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
N = 50000

time = np.arange(N) * Ts

# ===============================
# TRUE SYSTEM (3 states)
# ===============================

Phi_true = np.array([
    [1, Ts, 0],
    [0, 1,  Ts],
    [0, 0,  0]
])

Gamma = np.array([[0],
                  [0],
                  [1]])

H_true = np.array([[0, 0, 1]])

# ===============================
# KALMAN FILTER (4 states)
# ===============================

Phi_kf = np.array([
    [1, Ts, 0, 0],
    [0, 1,  Ts, 0],
    [0, 0,  phi, 0],
    [0, 0,  0, 1]
])

Q = np.diag([0, 0, sigma_wa**2, sigma_wb**2])
H_kf = np.array([[0, 0, 1, 1]])
R = np.array([[sigma_v**2]])

# ===============================
# Storage
# ===============================

x_true = np.zeros((3, N))
x_hat  = np.zeros((4, N))
P = np.eye(4)

# bias
b = b_true

# acceleration input
w1 = 2*np.pi*0.1      # 1 Hz
w2 = 2*np.pi*1      # 5 Hz

a_input = np.sin(w1*time) + 0.7*np.sin(w2*time)

# ===============================
# Simulation + KF
# ===============================

for k in range(N-1):

    # ---- True system ----
    x_true[:,k+1] = (
        Phi_true @ x_true[:,k] +
        (Gamma.flatten() * a_input[k])
    )

    # ===============================
    # Kalman Filter
    # ===============================

    # measurement: y = a + b + noise
    y = a_input[k] + b + sigma_v * np.random.randn()
    # b is fixed (-0.2), no variance term
    # a_input is deterministic (sinusoid), so no process noise here

    # PREDICTION STEP, both state and uncertianty covariance given previous samples
    x_pred = Phi_kf @ x_hat[:,k]            #x_hat from previous iteration
    P_pred = Phi_kf @ P @ Phi_kf.T + Q      #P from previoius iteration

    S = H_kf @ P_pred @ H_kf.T + R          # total uncertianty
    K = P_pred @ H_kf.T @ np.linalg.inv(S)  # kalman gain (model uncertianty/total uncerntianty)

    innovation = y - H_kf @ x_pred

    # update for state and covariance
    x_hat[:,k+1] = x_pred + (K @ innovation).flatten()  
    P = (np.eye(4) - K @ H_kf) @ P_pred                 #joseph form


# ===============================
# Plot Results
# ===============================



plt.figure(figsize=(12,10))
plt.suptitle("Exercise 2")
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
plt.plot(time, a_input, label='True a')
plt.plot(time, x_hat[2,:], '--', label='Estimated a')
plt.title('Acceleration')
plt.legend()
plt.grid()

# Bias
plt.subplot(4,1,4)
plt.plot(time, b_true*np.ones(N), label='True b')
plt.plot(time, x_hat[3,:], '--', label='Estimated b')
plt.title('Bias')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()