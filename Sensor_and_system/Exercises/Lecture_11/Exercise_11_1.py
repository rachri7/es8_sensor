import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. Simulation settings
# ============================================================

Ts = 0.01          # sample time
T_end = 10.0      # final simulation time
t = np.arange(0, T_end + Ts, Ts)
N = len(t)

np.random.seed(4)  # makes noise repeatable


# ============================================================
# 2. Continuous-time system
# ============================================================

# state transtion
A = np.array([
    [0.0, 1.0],
    [0.0, -0.015]
])

# scaling of input
B = np.array([
    [0.0],
    [0.5]
])


# We use Simple Euler discretization, drop limit use finite step
# when isolateing x[k+1] 
# x[k+1] = (I + Ts*A)x[k] + Ts*B*u[k]

Ad = np.eye(2) + Ts * A     #(I + Ts*A)
Bd = Ts * B                 # Ts*B

#input
u = np.cos(t) + 0.1 * np.sin(t / 2)     #Known deterministic velocity 


# ============================================================
# 3. Simulate true system
# ============================================================

x_true = np.zeros((N, 2))

# Use Simple Euler discretization
# # x[k+1] = (I + Ts*A)x[k] + Ts*B*u[k]
# tells us how our discrete evolve
for k in range(N - 1):
    x_true[k + 1] = Ad @ x_true[k] + Bd.flatten() * u[k]

y_true = x_true[:, 0]
v_true = x_true[:, 1]


# ============================================================
# 4. Add measurement noise
# ============================================================

var_y = 0.05
var_v = 0.1

std_y = np.sqrt(var_y)
std_v = np.sqrt(var_v)

y_meas = y_true + std_y * np.random.randn(N)
v_meas = v_true + std_v * np.random.randn(N)


# ============================================================
# 5. Complementary filter for position estimation
# ============================================================


# sets our cutoff frequency, larger tau means 
tau = 0.5

alpha1 = 2 * tau / Ts + 1
alpha0 = 1 - 2 * tau / Ts
beta = alpha1 - 1

y_comp = np.zeros(N)
y_comp[0] = y_meas[0]     # set first compute to same as 1st measured


for k in range(1, N):
    y_comp[k] = (
        -alpha0 * y_comp[k - 1]                         # is larger negative when tau bigger, 
        + beta * Ts / 2 * (v_meas[k] + v_meas[k - 1])   # integrates velocity measurement to position change, larger tau more influence on this
        + y_meas[k] + y_meas[k - 1]                     # smooth interpretation
    ) / alpha1                                          # ensure gain correct else drift

# becuase of the sample scale Ts and tau trust velocity integral for small fast steps
# only let position shift slowly
# ============================================================
# 6. Kalman filter for sensor fusion
# ============================================================

H = np.eye(2)

# Measurement Noise covariance
R = np.array([          # off diagonal zero the two independent
    [0.05, 0.0],        # 0.05 variance of position
    [0.0, 0.1]          # 0.1 variance of velocity
])

# Process noise covariance
Q = np.array([          # how much we trust dynamic model
    [1e-5, 0.0],        # both very small, means we think model accurate
    [0.0, 1e-4]         # becuase we basically say what Ad X + bd u is weight highly becuase this is low
])

x_hat = np.zeros((N, 2))    # initial posterior

# Initial estimate
x_pred = np.array([0.0, 0.0])

P = np.eye(2)   # Initial uncertianty covariance

I = np.eye(2)

for k in range(N):

    # Measurement vector
    z = np.array([y_meas[k], v_meas[k]])    # we recieve measurement, same model as before different filter

    # ----------------------------
    # Measurement update
    # ----------------------------

    S = H @ P @ H.T + R                     # total uncertianty: model uncertainty on measurement and measurement noise
    K = P @ H.T @ np.linalg.inv(S)          # Kalman gain: (model uncertianty / total uncertianty) if Larger = trust measurement

    x_update = x_pred + K @ (z - H @ x_pred)    # pred + innovation kalman scaled

    # Covariance update
    P = (I - K @ H) @ P @ (I - K @ H).T + K @ R @ K.T   # joseph form update uncertianty (decrease as get measurement)

    x_hat[k] = x_update

    # ----------------------------
    # Time update
    # ----------------------------

    if k < N - 1:
        x_pred = Ad @ x_update + Bd.flatten() * u[k]    # propagate estimate forward to make pred + determinstic known input scaled by bd
        P = Ad @ P @ Ad.T + Q                           # propagate uncertianty covarince forward plus proces noise (increase uncertianty)


y_kalman = x_hat[:, 0]
v_kalman = x_hat[:, 1]


# ============================================================
# 7. Plot 1: Complementary filter position estimate
# ============================================================

plt.figure(figsize=(10, 4))

plt.plot(t, y_true, "r", label="true y")
plt.plot(t, y_meas, "b.", markersize=2, label="measured y")
plt.plot(t, y_comp, "k", linewidth=2, label="complementary estimate")

plt.title("Complementary filter: position estimation")
plt.xlabel("time [s]")
plt.ylabel("y, measured y, estimated y")
plt.grid(True)
plt.legend(loc="best")
plt.tight_layout()


# ============================================================
# 8. Plot 2: Kalman filter sensor fusion
# ============================================================

plt.figure(figsize=(10, 7))

# Position subplot
plt.subplot(2, 1, 1)
plt.plot(t, y_true, "r", label="true y")
plt.plot(t, y_meas, "b.", markersize=2, label="measured y")
plt.plot(t, y_kalman, "k", linewidth=2, label="Kalman estimate")

plt.title("Kalman filter: position estimate")
plt.ylabel("y, measured y, estimated y")
plt.grid(True)
plt.legend(loc="best")


# Velocity subplot
plt.subplot(2, 1, 2)
plt.plot(t, v_true, "r", label="true v")
plt.plot(t, v_meas, "b.", markersize=2, label="measured v")
plt.plot(t, v_kalman, "k", linewidth=2, label="Kalman estimate")

plt.title("Kalman filter: velocity estimate")
plt.xlabel("time [s]")
plt.ylabel("v, measured v, estimated v")
plt.grid(True)
plt.legend(loc="best")

plt.tight_layout()
plt.show()


# ============================================================
# 9. Optional error calculation
# ============================================================

rmse_y_meas = np.sqrt(np.mean((y_meas - y_true) ** 2))
rmse_y_comp = np.sqrt(np.mean((y_comp - y_true) ** 2))
rmse_y_kalman = np.sqrt(np.mean((y_kalman - y_true) ** 2))

rmse_v_meas = np.sqrt(np.mean((v_meas - v_true) ** 2))
rmse_v_kalman = np.sqrt(np.mean((v_kalman - v_true) ** 2))

print("Position RMSE:")
print("Measured y:        ", rmse_y_meas)
print("Complementary y:   ", rmse_y_comp)
print("Kalman y:          ", rmse_y_kalman)

print()
print("Velocity RMSE:")
print("Measured v:        ", rmse_v_meas)
print("Kalman v:          ", rmse_v_kalman)