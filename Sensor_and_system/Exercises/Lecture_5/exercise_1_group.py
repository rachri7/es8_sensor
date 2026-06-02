import numpy as np
import matplotlib.pyplot as plt

np.random.seed(52)
# ===============================
# Parameters
# ===============================
Ts = 0.001          # sample period
phi = 0.9           # AR(1) coeffiecient
# standard deviations
sigma_wa = 0.1
sigma_wb = 0
sigma_v  = 0.1

b_true = -0.2              

N = 50000   # simulation length

# ===============================
# System Matrices
# ===============================
Phi = np.array([                                    # state transistion
    [1, Ts, 0, 0],                                  # position kinematics
    [0, 1,  Ts, 0],                                 # velocity kinematics
    [0, 0,  phi, 0],                                # acceleration AR(1)
    [0, 0,  0, 1]                                   # bais random walk
])

Q = np.diag([0, 0, sigma_wa**2, sigma_wb**2])       # process noise
H = np.array([[0, 0, 1, 1]])                        # measurement matrix
R = np.array([[sigma_v**2]])                        # measurment noise

# ===============================
# Storage
# ===============================
x_true = np.zeros((4, N))       # all true, position, velocity and acceleraton, bias
x_hat  = np.zeros((4, N))       # estimate kalman position, velocity, acceleraton and bias
P = np.eye(4)                   # intial uncertianty covariance (identity)

# initial bias
x_true[3,0] = b_true

# ===============================
# Simulation + Kalman Filter
# ===============================
for k in range(N-1):

    # ---- True system ----
    w = np.random.randn(4)                      # 4 gaussian noises
    w_scaled = np.array([0, 0,                  # process noise
                         sigma_wa * w[2],       # scale by std
                         sigma_wb * w[3]])

    x_true[:,k+1] = Phi @ x_true[:,k] + w_scaled    # evoluation of each state + process noise

    # ===============================
    # Kalman Filter
    # ===============================

    # Measurement
    v = sigma_v * np.random.randn()        
    y = H @ x_true[:,k] + v             # map state to measurement space and measurement noise

    # PREDICTION STEP, both state and uncertianty covariance given previous samples
    x_pred = Phi @ x_hat[:,k]           # transition previous estimate forward to be prediction
    P_pred = Phi @ P @ Phi.T + Q        # foward uncertianty coveriance + procces noise 

    S = H @ P_pred @ H.T + R                # total uncertianty
    K = P_pred @ H.T @ np.linalg.inv(S)     # kalman gain (model uncertianty/total uncerntianty)

    innovation = y - H @ x_pred

    # update for state and covariance
    x_hat[:,k+1] = x_pred + (K @ innovation).flatten()
    I = np.eye(4)
    P = (I - K @ H) @ P_pred @ (I - K @ H).T + K @ R @ K.T      #joseph form         
    

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