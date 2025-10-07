import numpy as np

# Fixed parameters (from your notebook)
DEFAULT_NUM_UAVS = 20
DEFAULT_DURATION = 50.0
DEFAULT_DT = 0.1
DEFAULT_U_I = 1.0
DEFAULT_R_I = 0.5
DEFAULT_WIND_SPEED = 0.0
DEFAULT_WIND_DIR = 0.0
DEFAULT_SEED = 42

def dynamics_model(state, control_input, wind_speed, wind_dir):
    x, y, psi = state
    u_i, r_i = control_input
    x_dot = u_i * np.cos(psi) + wind_speed * np.cos(wind_dir)
    y_dot = u_i * np.sin(psi) + wind_speed * np.sin(wind_dir)
    psi_dot = r_i
    return np.array([x_dot, y_dot, psi_dot], dtype=float)

def simulate(
    num_uavs=DEFAULT_NUM_UAVS,
    duration=DEFAULT_DURATION,
    dt=DEFAULT_DT,
    u_i=DEFAULT_U_I,
    r_i=DEFAULT_R_I,
    wind_speed=DEFAULT_WIND_SPEED,
    wind_dir=DEFAULT_WIND_DIR,
    seed=DEFAULT_SEED,
):
    rng = np.random.default_rng(seed)
    T = int(duration / dt)
    states = np.zeros((T, num_uavs, 3), dtype=float)  # [x, y, psi]
    # Random initial positions within area ~5x5, random orientation
    states[0, :, 0:2] = rng.random((num_uavs, 2)) * 5.0
    states[0, :, 2]   = rng.random(num_uavs) * 2 * np.pi

    for t in range(1, T):
        for j in range(num_uavs):
            s_prev = states[t-1, j]
            s_dot = dynamics_model(
                s_prev,
                np.array([u_i, r_i], dtype=float),
                wind_speed,
                wind_dir,
            )
            states[t, j] = s_prev + s_dot * dt

    return states  # shape: (T, N, 3)
