import numpy as np
import matplotlib.pyplot as plt

# ===============================
# Parameters
# ===============================
Ts = 0.001
phi = 0.9

sigma_wa = 0.1
sigma_wb = 0
sigma_v  = 0.1

b_true = -0.2

N = 20000   # simulation length

# ===============================
# System Matrices
# ===============================
Phi = np.array([
    [1, Ts, 0, 0],
    [0, 1,  Ts, 0],
    [0, 0,  phi, 0],
    [0, 0,  0, 1]
])

Q = np.diag([0, 0, sigma_wa**2, sigma_wb**2])
H = np.array([[0, 0, 1, 1]])
R = np.array([[sigma_v**2]])

# ===============================
# Storage
# ===============================
x_true = np.zeros((4, N))
x_hat  = np.zeros((4, N))
P      = np.eye(4)

# initial bias
x_true[3,0] = b_true

# ===============================
# Simulation + Kalman Filter
# ===============================
for k in range(N-1):

    # ---- True system ----
    w = np.random.randn(4)
    w_scaled = np.array([0, 0,
                         sigma_wa * w[2],
                         sigma_wb * w[3]])

    x_true[:,k+1] = Phi @ x_true[:,k] + w_scaled

    # Measurement
    v = sigma_v * np.random.randn()
    y = H @ x_true[:,k] + v

    # ===============================
    # Kalman Filter
    # ===============================

    # Prediction
    x_pred = Phi @ x_hat[:,k]
    P_pred = Phi @ P @ Phi.T + Q

    # Measurement update
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)

    innovation = y - H @ x_pred
    x_hat[:,k+1] = x_pred + (K @ innovation).flatten()

    P = (np.eye(4) - K @ H) @ P_pred

# ===============================
# Plot Results
# ===============================

time = np.arange(N) * Ts

plt.figure(figsize=(12,10))

# Position
plt.subplot(4,1,1)
plt.plot(time, x_true[0,:], label='True x')
plt.plot(time, x_hat[0,:], '--', label='Estimated x')
plt.title('Position')
plt.legend()
plt.grid()

# Velocity
plt.subplot(4,1,2)
plt.plot(time, x_true[1,:], label='True v')
plt.plot(time, x_hat[1,:], '--', label='Estimated v')
plt.title('Velocity')
plt.legend()
plt.grid()

# Acceleration 
plt.subplot(4,1,3)
plt.plot(time, x_true[2,:], label='True a')
plt.plot(time, x_hat[2,:], '--', label='Estimated a')
plt.title('Acceleration')
plt.legend()
plt.grid()

# Bias
plt.subplot(4,1,4)
plt.plot(time, x_true[3,:], label='True b')
plt.plot(time, x_hat[3,:], '--', label='Estimated b')
plt.title('Bias')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()