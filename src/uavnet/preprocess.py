import numpy as np

def scale_positions(states):
    """
    states: (T, N, 3) -> returns x_scaled (T, N, 2) with x,y scaled to [0,1]
    """
    x = states[..., :2].copy()
    for dim in (0, 1):
        lo = x[..., dim].min()
        hi = x[..., dim].max()
        ptp = (hi - lo) if (hi > lo) else 1.0
        x[..., dim] = (x[..., dim] - lo) / ptp
    return x

def compute_snr(x_scaled, radius=0.9, P_t=1.0, G=1.0, N0=1.0, alpha=0.5):
    """
    x_scaled: (T, N, 2) positions in [0,1]
    returns snr: (T, N)
    """
    T, N, _ = x_scaled.shape
    snr = np.zeros((T, N), dtype=float)
    for t in range(T):
        coords = x_scaled[t]  # (N,2)
        for i in range(N):
            d = np.linalg.norm(coords - coords[i], axis=1)
            d[i] = np.inf  # exclude self
            within = d <= radius
            if np.any(within):
                signal_powers = P_t * G * np.power(d[within], -alpha)
                snr[t, i] = signal_powers.mean() / N0
            else:
                j = int(np.argmin(d))
                snr[t, i] = (P_t * G * (d[j] ** -alpha)) / N0
    return snr

def scale_scalar_overall(arr):
    """
    arr: any numeric array -> scaled to [0,1] using global min/max
    """
    lo, hi = arr.min(), arr.max()
    ptp = (hi - lo) if (hi > lo) else 1.0
    return (arr - lo) / ptp
