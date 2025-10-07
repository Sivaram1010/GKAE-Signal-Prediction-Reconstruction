import argparse
from pathlib import Path
import numpy as np
import torch

from .simulate_uav import simulate
from .preprocess import scale_positions, compute_snr, scale_scalar_overall
from .build_graphs import build_graphs, make_loaders

def main():
    p = argparse.ArgumentParser(description="Build PyG dataset from UAV simulation")
    p.add_argument("--outdir", type=str, default="data/processed")
    p.add_argument("--num_uavs", type=int, default=20)
    p.add_argument("--duration", type=float, default=50.0)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--radius", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) simulate states (T,N,3) with fixed defaults matching your notebook
    states = simulate(
        num_uavs=args.num_uavs,
        duration=args.duration,
        dt=args.dt,
        seed=args.seed,
    )

    # 2) preprocess
    x_scaled = scale_positions(states)            # (T,N,2)
    snr      = compute_snr(x_scaled, radius=args.radius)  # (T,N)
    snr_s    = scale_scalar_overall(snr)          # scaled to [0,1]

    # 3) graphs -> PyG list
    pyg_list = build_graphs(x_scaled, snr_s, radius=args.radius)

    # 4) save artifacts
    np.savez_compressed(
        outdir / "uav_sim_interim.npz",
        states=states, x_scaled=x_scaled, snr=snr, snr_scaled=snr_s
    )
    torch.save(pyg_list, outdir / "pyg_graphs.pt")

    # 5) quick summary
    train, val = make_loaders(pyg_list)
    print(f"Built {len(pyg_list)} graphs | train batches: {len(train)}, val batches: {len(val)}")
    print(f"Wrote: {outdir / 'uav_sim_interim.npz'}")
    print(f"Wrote: {outdir / 'pyg_graphs.pt'}")

if __name__ == "__main__":
    main()
