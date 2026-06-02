import numpy as np
import matplotlib.pyplot as plt

np.random.seed(4)
# ===============================
# Parameters
# ===============================
Ts = 0.001
phi = 0.9

# have made the variables indepedent per axis cause the sensors may differ
sigma_wa_x = 0.1;  sigma_wa_y = 0.1
sigma_wb_x = 0;    sigma_wb_y = 0
sigma_v_x  = 0.1;  sigma_v_y  = 0.1

b_true_x = -0.2
b_true_y = -0.2
N = 50000

time = np.arange(N) * Ts

# ===============================
# TRUE SYSTEM
# ===============================

Phi_true_1d = np.array([
    [1, Ts, 0],
    [0, 1,  Ts],
    [0, 0,  0]
])

# Expanded to 6x6 block diagonal for 2D [x, vx, ax, y, vy, ay]
# Off-diagonal zeros: x and y axes evolve independently
Phi_true = np.block([
    [Phi_true_1d, np.zeros((3,3))],
    [np.zeros((3,3)), Phi_true_1d]
])

# Expand to 2x6 for 2D and all variables to each axis
Gamma = np.array([[0, 0],
                  [0, 0],
                  [1, 0],
                  [0, 0],
                  [0, 0],
                  [0, 1]])

# ===============================
# KALMAN FILTER
# ===============================

Phi_kf_1d = np.array([
    [1, Ts, 0,   0],
    [0, 1,  Ts,  0],
    [0, 0,  phi, 0],
    [0, 0,  0,   1]
])

# Expanded to 8x8 block diagonal for 2D [x, vx, ax, bx, y, vy, ay, by]
# Off-diagonal zeros: axes are independent, no cross-axis transition
Phi_kf = np.block([
    [Phi_kf_1d, np.zeros((4,4))],
    [np.zeros((4,4)), Phi_kf_1d]
])

# just the same expanded
Q = np.diag([0, 0, sigma_wa_x**2, sigma_wb_x**2,
             0, 0, sigma_wa_y**2, sigma_wb_y**2])

# Expanded to 2x8: one measurement row per axis
H_kf = np.array([
    [0, 0, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 1]
])

R = np.diag([sigma_v_x**2, sigma_v_y**2])       # includes to measurement noise, cause two sensors
# (also 2states now)

# ===============================
# Storage
# ===============================

x_true = np.zeros((6, N))
x_hat  = np.zeros((8, N))
P = np.eye(8)

# ===============================
# Acceleration inputs
# ===============================

w1 = 2*np.pi*0.1
w2 = 2*np.pi*1

a_input_x = np.sin(w1*time) + 0.7*np.sin(w2*time)       # determinstic input (known)
a_input_y = np.sin(w1*time) + 0.7*np.sin(w2*time)       # determinstic input (known)

# ===============================
# Simulation + KF
# ===============================

for k in range(N-1):

    # ---- True system ----
    # state transtion and input
    x_true[:,k+1] = Phi_true @ x_true[:,k] + Gamma @ np.array([a_input_x[k], a_input_y[k]])


    # ---- Kalman Filter ----

    # Measurement: y = a + b + noise
    y = np.array([
        a_input_x[k] + b_true_x + sigma_v_x * np.random.randn(),
        a_input_y[k] + b_true_y + sigma_v_y * np.random.randn()
    ])

    # PREDICTION STEP, both state and uncertianty covariance given previous samples
    x_pred = Phi_kf @ x_hat[:,k]                #x_hat from previous iteration
    P_pred = Phi_kf @ P @ Phi_kf.T + Q          #P from previoius iteration

    S = H_kf @ P_pred @ H_kf.T + R          # total uncertianty
    K = P_pred @ H_kf.T @ np.linalg.inv(S)  # kalman gain (model uncertianty/total uncerntianty)

    innovation = y - H_kf @ x_pred

    # update for state and covariance
    x_hat[:,k+1] = x_pred + K @ innovation
    I = np.eye(8)
    P = (I - K @ H_kf) @ P_pred @ (I - K @ H_kf).T + K @ R @ K.T      #joseph form        

# ===============================
# Plot Results
# ===============================

fig, axes = plt.subplots(4, 2, figsize=(14, 12))
plt.suptitle("Exercise 4 --- 2D Kalman Filter")

for col, (lbl, a_in, b_tr, ti, hi) in enumerate(zip(
    ['X-axis', 'Y-axis'],
    [a_input_x, a_input_y],
    [b_true_x,  b_true_y],
    [[0,1,2,None], [3,4,5,None]],
    [[0,1,2,3],    [4,5,6,7]]
)):
    for row, title in enumerate(['Position', 'Velocity', 'Acceleration', 'Bias']):
        ax = axes[row, col]
        true_sig = a_in if row == 2 else (b_tr * np.ones(N) if row == 3 else x_true[ti[row],:])
        ax.plot(time, true_sig, label='True')
        ax.plot(time, x_hat[hi[row],:], '--', label='Estimated')
        ax.set_title(f'{lbl} --- {title}')
        ax.legend(); ax.grid()

plt.tight_layout()
plt.show()