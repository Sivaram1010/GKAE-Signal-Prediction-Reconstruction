from pathlib import Path
import argparse
import numpy as np

def mse(a, b): return float(np.mean((a - b) ** 2))
def rmse(a, b): return float(np.sqrt(mse(a, b)))
def mae(a, b): return float(np.mean(np.abs(a - b)))

def main():
    ap = argparse.ArgumentParser(description="Evaluate GAE reconstruction and export Koopman inputs")
    ap.add_argument("--data_dir", type=str, default="data/processed/run1",
                    help="Folder that contains gae_outputs.npz from training")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    npz_path = data_dir / "gae_outputs.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing {npz_path}. Run train_gae first.")

    Z = np.load(npz_path)
    real = Z["real"]       # (T, N)
    pred = Z["pred"]       # (T, N)
    z     = Z["z"]         # (T, latent)
    z_s   = Z["z_scaled"]  # (T, latent)
    X_train = Z["X_train"] # (300, latent)
    X_test  = Z["X_test"]  # (201, latent) note overlap at 299 per your split

    # Basic metrics (on the 500-sample val set)
    metrics = {
        "MSE":  mse(real, pred),
        "RMSE": rmse(real, pred),
        "MAE":  mae(real, pred),
    }

    # Save metrics + arrays for downstream Koopman
    out_metrics = data_dir / "gae_eval_metrics.txt"
    with out_metrics.open("w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v:.8f}\n")

    # Also save convenient CSVs for quick plots/inspection
    np.savetxt(data_dir / "real.csv", real, delimiter=",")
    np.savetxt(data_dir / "pred.csv", pred, delimiter=",")
    np.savetxt(data_dir / "z.csv", z, delimiter=",")
    np.savetxt(data_dir / "z_scaled.csv", z_s, delimiter=",")
    np.savetxt(data_dir / "X_train.csv", X_train, delimiter=",")
    np.savetxt(data_dir / "X_test.csv",  X_test, delimiter=",")

    print("Saved:")
    print(f" - {out_metrics}")
    print(f" - {data_dir/'real.csv'}")
    print(f" - {data_dir/'pred.csv'}")
    print(f" - {data_dir/'z.csv'}")
    print(f" - {data_dir/'z_scaled.csv'}")
    print(f" - {data_dir/'X_train.csv'}")
    print(f" - {data_dir/'X_test.csv'}")
    print("Metrics:", metrics)

if __name__ == "__main__":
    main()
