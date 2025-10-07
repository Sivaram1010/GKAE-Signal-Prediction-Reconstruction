from pathlib import Path
import argparse
import numpy as np
import torch
import torch.nn.functional as F

from .koopman_train_user import koopmanAE   # reuse the class definition
from .models import Autoencoder, NUM_UAVS

def prediction_loop(Xinput, Xtarget, model, autoenc, start, pred_steps,
                    min_val, max_val, snr_min, snr_ptp, device):
    """
    Xinput/Xtarget: torch tensors shaped [T, 1, m, 1] with m=latent_dim(=20).
    model: trained koopmanAE; autoenc: trained Autoencoder (decoder used).
    """
    m, n = Xinput.shape[2], 1

    snapshots_pred = []
    snapshots_truth = []

    # init latent (scaled)
    init = Xinput[start].float().to(device)         # [1,1,m,1]
    z = model.encoder(init)                         # [1,1,bottle]

    for j in range(pred_steps):
        z = model.dynamics(z)                       # propagate
        x_pred = model.decoder(z)                   # [1,1,m,1] (scaled latent)
        # ground truth (scaled latent) from target
        target_temp = Xtarget[start + j].cpu().numpy().reshape(m, n)
        pred_temp   = x_pred.detach().cpu().numpy().reshape(m, n)
        snapshots_pred.append(pred_temp)
        snapshots_truth.append(target_temp)

    snapshots_pred  = np.array(snapshots_pred).reshape(pred_steps, m)   # scaled latent
    snapshots_truth = np.array(snapshots_truth).reshape(pred_steps, m)  # scaled latent

    # rescale latent back to original z using min/max saved by preproc
    original_pred  = snapshots_pred  * (max_val - min_val) + min_val    # (steps, m)
    original_truth = snapshots_truth * (max_val - min_val) + min_val

    # decode to node space using the GAE decoder
    pred_values = []
    real_values = []
    unscaled_pred = []   # keep decoder outputs before snr rescale (for debugging)

    for t in range(pred_steps):
        # predicted
        pred_tensor = torch.from_numpy(original_pred[t]).float().to(device)  # [m]
        dec_pred = autoenc.decoder(pred_tensor.unsqueeze(0))  # -> [1, NUM_UAVS, 1]
        dec_pred_np = dec_pred.detach().cpu().numpy().reshape(NUM_UAVS)
        unscaled_pred.append(dec_pred_np.copy())
        # map from [0,1] back to SNR using snr_min/ptp
        pred_values.append(dec_pred_np * snr_ptp + snr_min)

        # truth (decode latent truth as well)
        truth_tensor = torch.from_numpy(original_truth[t]).float().to(device)
        dec_truth = autoenc.decoder(truth_tensor.unsqueeze(0))  # [1, NUM_UAVS, 1]
        dec_truth_np = dec_truth.detach().cpu().numpy().reshape(NUM_UAVS)
        real_values.append(dec_truth_np * snr_ptp + snr_min)

    pred_values = np.array(pred_values)  # (steps, NUM_UAVS)
    real_values = np.array(real_values)  # (steps, NUM_UAVS)
    return pred_values, real_values, unscaled_pred

def main():
    ap = argparse.ArgumentParser(description="Evaluate Koopman-AE rollout vs ground truth")
    ap.add_argument("--data_dir", type=str, default="data/processed/run1",
                    help="Folder containing gae_outputs.npz, koopman_preproc.pt, koopman_ae_user.pt, gae_model.pt")
    ap.add_argument("--pred_length", type=int, default=1)
    ap.add_argument("--window_count", type=int, default=200, help="how many 1-step windows to evaluate")
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    data_dir = Path(args.data_dir)

    # load preproc pack
    pack = torch.load(data_dir / "koopman_preproc.pt", weights_only=False)
    Xtest = pack["Xtest"]    # [201,1,m,1] scaled latent
    TL    = int(pack["TL"])
    m     = int(pack["m"])   # was set to num_uavs in preproc, but equals latent_dim (=20) here
    bottle= int(pack["bottle"])
    min_val = pack["min_val"]
    max_val = pack["max_val"]

    # build Xin, Xtarget
    Xin     = Xtest[:-1]   # [200,1,m,1]
    Xtarget = Xtest[1:]    # [200,1,m,1]

    # snr stats for rescaling node space back to original SNR units
    # pull from simulation outputs
    sim = np.load(data_dir / "uav_sim_interim.npz")
    snr = sim["snr"]  # (T, NUM_UAVS)
    snr_min = float(np.min(snr))
    snr_max = float(np.max(snr))
    snr_ptp = snr_max - snr_min

    # load Koopman model
    from .koopman_train_user import koopmanAE
    model = koopmanAE(m=m, n=1, bottle=bottle, steps=TL, steps_back=TL, alpha=1, init_scale=1.0).to(device)
    model.load_state_dict(torch.load(data_dir / "koopman_ae_user.pt", map_location=device, weights_only=False))
    model.eval()

    # load Autoencoder (to use its decoder)
    autoenc = Autoencoder(in_features=1, latent_dim=m, num_uavs=NUM_UAVS).to(device)
    autoenc.load_state_dict(torch.load(data_dir / "gae_model.pt", map_location=device, weights_only=False))
    autoenc.eval()

    # run rolling windows
    pred_values_loop = []
    real_values_loop = []

    steps = min(args.window_count, Xin.shape[0])
    for i in range(0, steps, args.pred_length):
        pred_vals, real_vals, _ = prediction_loop(
            Xin, Xtarget, model, autoenc, start=i, pred_steps=args.pred_length,
            min_val=min_val, max_val=max_val,
            snr_min=snr_min, snr_ptp=snr_ptp, device=device
        )
        pred_values_loop.append(pred_vals)  # (pred_length, N)
        # use GAEs' 'real' from saved outputs to match your reference (optional):
        # or keep real_vals decoded above:
        real_values_loop.append(real_vals)

    pred_values_loop = np.concatenate(pred_values_loop, axis=0)  # (steps, N)
    real_values_loop = np.concatenate(real_values_loop, axis=0)  # (steps, N)

    # metrics
    pred_t = torch.tensor(pred_values_loop)
    real_t = torch.tensor(real_values_loop)

    mae  = F.l1_loss(pred_t, real_t, reduction='mean').item()
    rmse = torch.sqrt(F.mse_loss(pred_t, real_t, reduction='mean')).item()

    # sMAPE
    numerator = torch.abs(pred_t - real_t)
    denominator = (torch.abs(pred_t) + torch.abs(real_t)) / 2 + 1e-6
    smape = (numerator / denominator).mean().item() * 100.0

    print(f"RMSE: {rmse:.6f}")
    print(f"MAE:  {mae:.6f}")
    print(f"sMAPE: {smape:.3f}%")

    # save arrays for inspection
    np.savez_compressed(data_dir / "koopman_eval_outputs.npz",
                        pred=pred_values_loop, real=real_values_loop,
                        rmse=rmse, mae=mae, smape=smape)
    print(f"Saved: {data_dir/'koopman_eval_outputs.npz'}")

if __name__ == "__main__":
    main()
