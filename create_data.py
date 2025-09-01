import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
# Define simulation parameters
dt = 0.1  # Time step in seconds
duration = 50  # Duration of the simulation
T = int(duration / dt)  # Number of time steps
num_uavs = 20  # Number of UAVs

u_i = 1  # Random velocity
r_i = 0.5
v_iw = 0# Wind speed
theta_iw = 0  # Wind direction

# Initialize arrays
states = np.zeros((T, num_uavs, 3))  # [x, y, psi] for each UAV

# Random initial positions and orientations
states[0, :, 0:2] = np.random.rand(num_uavs, 2)*5  # Random initial positions within the area
states[0, :, 2] = np.random.rand(num_uavs) * 2 * np.pi  # Random initial orientations

# Define the dynamics model function
def dynamics_model(state, control_input, wind_speed, wind_direction):
    x, y, psi = state
    u_i, r_i = control_input
    v_iw = wind_speed
    theta_iw = wind_direction
    
    x_dot = u_i * np.cos(psi) + v_iw * np.cos(theta_iw)
    y_dot = u_i * np.sin(psi) + v_iw * np.sin(theta_iw)
    psi_dot = r_i
    
    return np.array([x_dot, y_dot, psi_dot])

# Simulate UAV trajectories
for i in range(1, T):
    for j in range(num_uavs):
        state = states[i-1, j]
        control_input = np.array([u_i, r_i])
        state_dot = dynamics_model(state, control_input, v_iw, theta_iw)
        states[i, j] = state + state_dot * dt
        
#         # Add periodic boundary conditions (if required)
#         states[i, j, 0:2] = np.mod(states[i, j, 0:2], 100)

# Plot UAV trajectories
plt.figure(figsize=(10, 8))
for j in range(num_uavs):
    plt.plot(states[:200, j, 0], states[:200, j, 1], label=f"UAV {j+1}")
plt.title("Multi-UAV Trajectory with Dynamics Model")
plt.xlabel('X position (m)')
plt.ylabel('Y position (m)')
plt.legend()
plt.grid(True)
plt.show()
