# SPDX-License-Identifier: GPL-3.0-only
# This file incorporates and modifies code inspired by erichson/koopmanAE (GPL-3.0).
# Source: https://github.com/erichson/koopmanAE

from pathlib import Path
import argparse
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# ----- helpers -----
def gaussian_init_(n_units, std=1.0):
    # N(0, std/n_units)
    sampler = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([std / n_units]))
    Omega = sampler.sample((n_units, n_units))[..., 0]
    return Omega

# ----- models -----
class encoderNet(nn.Module):
    def __init__(self, m, n, bottle, ALPHA=1):
        super().__init__()
        self.N = m * n
        self.tanh = nn.Tanh()
        self.fc1 = nn.Linear(self.N, 16)
        self.fc2 = nn.Linear(16*ALPHA, 16*ALPHA)
        self.fc3 = nn.Linear(16*ALPHA, 16*ALPHA)
        self.fc4 = nn.Linear(16*ALPHA, 16*ALPHA)
        self.fc5 = nn.Linear(16*ALPHA, 16*ALPHA)
        self.fc6 = nn.Linear(16*ALPHA, bottle)

        for m_ in self.modules():
            if isinstance(m_, nn.Linear):
                nn.init.xavier_normal_(m_.weight)
                if m_.bias is not None:
                    nn.init.constant_(m_.bias, 0.0)

    def forward(self, x):
        # x: [B, 1, m, n] -> flatten last two dims
        x = x.view(-1, 1, self.N)
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = self.tanh(self.fc3(x))
        x = self.tanh(self.fc4(x))
        x = self.tanh(self.fc5(x))
        x = self.fc6(x)  # [B, 1, bottle]
        return x

class decoderNet(nn.Module):
    def __init__(self, m, n, bottle, ALPHA=1):
        super().__init__()
        self.m, self.n, self.bottle = m, n, bottle
        self.tanh = nn.Tanh()
        self.fc1 = nn.Linear(bottle, 16*ALPHA)
        self.fc2 = nn.Linear(16*ALPHA, 16*ALPHA)
        self.fc3 = nn.Linear(16*ALPHA, 16*ALPHA)
        self.fc4 = nn.Linear(16*ALPHA, 16*ALPHA)
        self.fc5 = nn.Linear(16*ALPHA, 16*ALPHA)
        self.fc6 = nn.Linear(16, m*n)

        for m_ in self.modules():
            if isinstance(m_, nn.Linear):
                nn.init.xavier_normal_(m_.weight)
                if m_.bias is not None:
                    nn.init.constant_(m_.bias, 0.0)

    def forward(self, x):
        # x: [B, 1, bottle]
        x = x.view(-1, 1, self.bottle)
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = self.tanh(self.fc3(x))
        x = self.tanh(self.fc4(x))
        x = self.tanh(self.fc5(x))
        x = self.tanh(self.fc6(x))      # [B, 1, m*n]
        x = x.view(-1, 1, self.m, self.n)
        return x

class dynamics(nn.Module):
    def __init__(self, bottle, init_scale):
        super().__init__()
        self.dynamics = nn.Linear(bottle, bottle, bias=False)
        self.dynamics.weight.data = gaussian_init_(bottle, std=1.0)
        # orthogonalize with SVD (torch.linalg.svd is the modern API)
        U, _, Vh = torch.linalg.svd(self.dynamics.weight.data, full_matrices=False)
        self.dynamics.weight.data = (U @ Vh) * init_scale

    def forward(self, x):  # x: [B, 1, bottle]
        return self.dynamics(x)

class dynamics_back(nn.Module):
    def __init__(self, bottle, omega: dynamics):
        super().__init__()
        self.dynamics = nn.Linear(bottle, bottle, bias=False)
        self.dynamics.weight.data = torch.pinverse(omega.dynamics.weight.data.t())

    def forward(self, x):
        return self.dynamics(x)

class koopmanAE(nn.Module):
    def __init__(self, m, n, bottle, steps, steps_back, alpha=1, init_scale=1.0):
        super().__init__()
        self.steps = steps
        self.steps_back = steps_back
        self.encoder = encoderNet(m, n, bottle, ALPHA=alpha)
        self.dynamics = dynamics(bottle, init_scale)
        self.backdynamics = dynamics_back(bottle, self.dynamics)
        self.decoder = decoderNet(m, n, bottle, ALPHA=alpha)

    def forward(self, x, mode='forward'):
        # x: [B, 1, m, n]
        out, out_back = [], []
        z = self.encoder(x).contiguous()  # [B,1,bottle]
        q = z.contiguous()

        if mode == 'forward':
            for _ in range(self.steps):
                q = self.dynamics(q)
                out.append(self.decoder(q))
            out.append(self.decoder(z))
            return out, out_back

        if mode == 'backward':
            for _ in range(self.steps_back):
                q = self.backdynamics(q)
                out_back.append(self.decoder(q))
            out_back.append(self.decoder(z))
            return out, out_back

# ----- training -----
def train(model, train_loader, lr, weight_decay, lamb, num_epochs,
          learning_rate_change, epoch_update, nu, eta, backward, steps, steps_back, gradclip, device):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss().to(device)
    loss_hist = []

    def lr_scheduler(optimizer, epoch, lr_decay_rate=0.8, decayEpoch=[]):
        if epoch in decayEpoch:
            for pg in optimizer.param_groups:
                pg['lr'] *= lr_decay_rate
        return optimizer

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        progress = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}", unit="batch")
        for _, data_list in progress:
            model.train()
            # forward mode
            out, out_back = model(data_list[0].to(device), mode='forward')

            # forward losses across predicted steps
            loss_fwd = 0.0
            for k in range(steps):
                loss_fwd = loss_fwd + criterion(out[k], data_list[k+1].to(device))

            # reconstruction loss / identity
            loss_identity = criterion(out[-1], data_list[0].to(device)) * steps

            loss_bwd = 0.0
            loss_consist = 0.0
            if backward == 1:
                # backward consistency
                out_fwd, out_back = model(data_list[-1].to(device), mode='backward')
                for k in range(steps_back):
                    loss_bwd = loss_bwd + criterion(out_back[k], data_list[::-1][k+1].to(device))

                A = model.dynamics.dynamics.weight
                B = model.backdynamics.dynamics.weight
                K = A.shape[-1]
                eye = torch.eye

                for k in range(1, K + 1):
                    As1 = A[:, :k]
                    Bs1 = B[:k, :]
                    As2 = A[:k, :]
                    Bs2 = B[:, :k]
                    Ik = eye(k, device=A.device, dtype=A.dtype)
                    loss_consist = loss_consist + (
                        torch.sum((Bs1 @ As1 - Ik) ** 2) +
                        torch.sum((As2 @ Bs2 - Ik) ** 2)
                    ) / (2.0 * k)

            loss = loss_fwd + lamb * loss_identity + nu * loss_bwd + eta * loss_consist

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradclip)
            opt.step()

            epoch_loss += float(loss.item())
            progress.set_postfix({'Loss': f"{float(loss.item()):.6f}"})

        lr_scheduler(opt, epoch, lr_decay_rate=learning_rate_change, decayEpoch=epoch_update)
        loss_hist.append(epoch_loss / max(1, len(train_loader)))

        # eigenvalues print (optional)
        with torch.no_grad():
            w = torch.linalg.eigvals(model.dynamics.dynamics.weight.data).cpu().numpy()
            print("eig|A|:", np.abs(w))

    return model, opt, loss_hist

# ----- dataset make -----
def make_train_loader(Xtrain: torch.Tensor, TL: int, batch_size: int = 16, shuffle=True):
    """
    Build sliding-window dataset list: [X[t], X[t+1], ..., X[t+TL]]
    Matches your notebook logic.
    """
    # Xtrain: [T_train, 1, m, n]
    trainDat = []
    start = 0
    for i in np.arange(TL, -1, -1):
        if i == 0:
            trainDat.append(Xtrain[start:].float())
        else:
            trainDat.append(Xtrain[start:-i].float())
        start += 1
    train_dataset = TensorDataset(*trainDat)
    return DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle)

def main():
    ap = argparse.ArgumentParser(description="Train Koopman AE (user-style)")
    ap.add_argument("--data_dir", type=str, default="data/processed/run1",
                    help="Folder containing koopman_preproc.pt")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--learning_rate_change", type=float, default=0.2)
    ap.add_argument("--epoch_update", type=int, nargs='*', default=[300, 350])
    ap.add_argument("--backward", type=int, default=0)
    ap.add_argument("--gradclip", type=float, default=0.05)
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    device = torch.device(args.device)

    # load preproc tensors
    pack = torch.load(Path(args.data_dir) / "koopman_preproc.pt", weights_only=False)
    Xtrain = pack["Xtrain"].to(device)  # [300,1,m,1]
    TL     = int(pack["TL"])
    m      = int(pack["m"])
    n      = int(pack["n"])
    bottle = int(pack["bottle"])

    # loader
    train_loader = make_train_loader(Xtrain, TL=TL, batch_size=args.batch_size, shuffle=True)

    # model
    steps = TL
    steps_back = TL
    model = koopmanAE(m, n, bottle, steps, steps_back, alpha=1, init_scale=1.0).to(device)

    # train
    model, opt, loss_hist = train(
        model, train_loader,
        lr=args.lr, weight_decay=args.weight_decay,
        lamb=1.0, num_epochs=args.epochs,
        learning_rate_change=args.learning_rate_change,
        epoch_update=args.epoch_update,
        nu=0.0, eta=0.0, backward=args.backward,
        steps=steps, steps_back=steps_back,
        gradclip=args.gradclip, device=device
    )

    # save
    out_model = Path(args.data_dir) / "koopman_ae_user.pt"
    out_loss  = Path(args.data_dir) / "koopman_ae_loss.npy"
    torch.save(model.state_dict(), out_model)
    np.save(out_loss, np.array(loss_hist, dtype=float))
    print(f"Saved model: {out_model}")
    print(f"Saved loss:  {out_loss}")

if __name__ == "__main__":
    main()
