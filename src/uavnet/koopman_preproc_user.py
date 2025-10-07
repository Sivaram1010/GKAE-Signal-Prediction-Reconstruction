from pathlib import Path
import argparse
import numpy as np
import torch

def add_channels(X: np.ndarray):
    # your helper
    if len(X.shape) == 2:
        return X.reshape(X.shape[0], 1, X.shape[1], 1)
    elif len(X.shape) == 3:
        return X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])
    else:
        raise ValueError("dimensional error")

def main():
    ap = argparse.ArgumentParser(description="Preprocess latents for Koopman AE (user logic)")
    ap.add_argument("--data_dir", type=str, default="data/processed/run1",
                    help="Folder with gae_outputs.npz")
    ap.add_argument("--TL", type=int, default=50, help="enforce linearity window")
    ap.add_argument("--num_uavs", type=int, default=20, help="N (m in your code)")
    ap.add_argument("--bottle", type=int, default=8, help="latent width for koopmanAE")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    npz = np.load(data_dir / "gae_outputs.npz")

    # === your exact logic ===
    z = npz["z"]                     # (500, 20)
    min_val, max_val = np.min(z), np.max(z)
    scaledz = (z - min_val) / (max_val - min_val + 1e-12)

    X_train = scaledz[0:300]
    X_test  = scaledz[299:500]

    # add channels (your function)
    Xtrain = add_channels(X_train)
    Xtest  = add_channels(X_test)

    # to torch tensors (float + contiguous)
    Xtrain_t = torch.from_numpy(Xtrain).float().contiguous()
    Xtest_t  = torch.from_numpy(Xtest).float().contiguous()

    # save everything needed for training
    out = {
        "Xtrain": Xtrain_t,
        "Xtest":  Xtest_t,
        "TL":     args.TL,
        "m":      args.num_uavs,
        "n":      1,
        "bottle": args.bottle,
        "min_val": float(min_val),
        "max_val": float(max_val),
    }
    torch.save(out, data_dir / "koopman_preproc.pt")
    print(f"Saved: {data_dir/'koopman_preproc.pt'}")
    print(f"Shapes: Xtrain {tuple(Xtrain_t.shape)}, Xtest {tuple(Xtest_t.shape)}, TL={args.TL}")

if __name__ == "__main__":
    main()
