import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# -----------------------------
# System matrices
# -----------------------------
A = np.array([          # state transitinon matrix (how state evolve without process noise)
    [0.7360, 0.2560],
    [0.3200, 0.6480]
])

C = np.array([          # measurement matrix: map state to measurement
    [0.5, 0.5],
    [0.0, 1.0]
])

Q = 0.4 * np.eye(2)     # process noise
R = 0.2 * np.eye(2)     # measurement noise

# -----------------------------
# Simulation settings
# -----------------------------
N = 100
rng = np.random.default_rng(1)

# True state
x = np.zeros((N + 1, 2))
x[0] = np.array([2.0, -1.0])

# Measurements
# NaN means no measurement at that time
y = np.full((N + 1, 2), np.nan)

# -----------------------------
# Simulate the true system
# -----------------------------
for k in range(N + 1):

    # Measurement noise
    v = rng.multivariate_normal(mean=np.zeros(2), cov=R)

    # Full measurement, before applying the sampling schedule (if all where measured)
    y_full = C @ x[k] + v

    # Only take measurement out when, specficicaly either y1,y2 when sensor reads

    # Sensor 1 gives y1 at k = 0, 5, 10, ... 
    if k % 5 == 0:
        y[k, 0] = y_full[0]

    # Sensor 2 gives y2 at k = 1, 6, 11, ...
    if k % 5 == 1:
        y[k, 1] = y_full[1]

    # Propagate true state by use of state transition matrix and process noise
    if k < N:
        w = rng.multivariate_normal(mean=np.zeros(2), cov=Q)
        x[k + 1] = A @ x[k] + w


# -----------------------------
# Kalman filter initialization
# -----------------------------
xhat_pred = np.array([0.0, 0.0])        # predict both state 0 (k-1)
P_pred = np.eye(2)                      # uncertianty covariance indetity both equal uncertian

xhat = np.zeros((N + 1, 2))             # all estimated after curent sample (k)
P_store = np.zeros((N + 1, 2, 2))       # all uncertianty covariances

# K_store[k, :, 0] stores gain for sensor 1
# K_store[k, :, 1] stores gain for sensor 2
# NaN means no Kalman gain at that time
K_store = np.full((N + 1, 2, 2), np.nan)


# -----------------------------
# Kalman filter
# -----------------------------
for k in range(N + 1):

    # By default, no measurement update
    # else overwritten by either of the two sensors
    xhat_update = xhat_pred.copy()
    P_update = P_pred.copy()
    # if no measurement just progate trough state transition (A)
    # uncertianty shuold increase cause we just add process noise to each sample 
    # and progate through state transition (A) 

    # -------------------------
    # Sensor 1 update
    # -------------------------
    if k % 5 == 0:

        # use 1st equation for both state equation and measurement equation (Or what)
        H = C[0:1, :]   # H = [0.5 0.5]
        Rk = R[0, 0]    # 0.2 variance for 1st sensor
        zk = y[k, 0]    # measured for 1st sensor

        # Innovation
        innovation = zk - (H @ xhat_pred).item()    #state prediction projectate on measurement space (measured - predicted)

        # Innovation covariance
        S = (H @ P_pred @ H.T + Rk).item()          # total uncertianty (model uncertianty and Measurement noise)

        # Kalman gain, is 2 X 1 dim, meaning scale correction of both x1 and x2
        K = (P_pred @ H.T) / S                      # find scale of uncertianty from model, Higher trust measurement more

        # State update
        xhat_update = xhat_pred + K[:, 0] * innovation  # estimate of state based of pred and innovation scaling

        # Covariance update, Joseph form
        I = np.eye(2)
        P_update = (
            (I - K @ H) @ P_pred @ (I - K @ H).T
            + K @ np.array([[Rk]]) @ K.T
        )

        # Store Kalman gain
        K_store[k, :, 0] = K[:, 0]

    # -------------------------
    # Sensor 2 update
    # -------------------------
    elif k % 5 == 1:

        # same principle as sensor 1 just now use state and measurement equation 2nd
        H = C[1:2, :]   # H = [0 1]
        Rk = R[1, 1]
        zk = y[k, 1]

        # Innovation
        innovation = zk - (H @ xhat_pred).item()

        # Innovation covariance
        S = (H @ P_pred @ H.T + Rk).item()

        # Kalman gain
        K = (P_pred @ H.T) / S

        # State update
        xhat_update = xhat_pred + K[:, 0] * innovation

        # Covariance update, Joseph form
        I = np.eye(2)
        P_update = (
            (I - K @ H) @ P_pred @ (I - K @ H).T
            + K @ np.array([[Rk]]) @ K.T
        )

        # Store Kalman gain
        K_store[k, :, 1] = K[:, 0]

    # Store filtered estimate and covariance
    xhat[k] = xhat_update
    P_store[k] = P_update

    # -------------------------
    # Prediction for next time step
    # -------------------------
    xhat_pred = A @ xhat_update
    P_pred = A @ P_update @ A.T + Q


# -----------------------------
# Estimated sensor outputs
# -----------------------------
yhat = xhat @ C.T               # we use C measurement matrix on all estimation x
y_true_without_noise = x @ C.T  # just purely state transition + process noise and using measurement matrix

k_values = np.arange(N + 1)


############## What see in results ################

# ---- Covariance plot ---
# It can be clearly observed that PP rises between measurements as only process noise Q is added each step,
# and falls whenever a sensor fires as the new measurement reduces uncertainty.

# ---- Kalman Gain Plot ----
# Sensor 1 first 4 samples after last measurement meaning higher uncertianty seen covariance plot
# Therefore high Kalman gain for both
# Sensor 2 is right after sensor 1 reading as we just got reading uncertianty is lower
# Means more certian about model so kalman gain not as high

# ---- result plot ----
# We can clearly also see this over on the plot vhere nudge very much towards y1 measurement for both x1 and x2
# while dont do it as much for the y2 readings still alot for x2 but not much for x1



# -----------------------------
# Plot true states, measurements, and estimates
# -----------------------------

# sensor firing times
k_s1 = k_values[k_values % 5 == 0]
k_s2 = k_values[k_values % 5 == 1]

# ================================================================
# Plot 1: States and measurements
# ================================================================
fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
fig.suptitle("Kalman filter — state estimates and sensor outputs", fontsize=13)
 
# helper: add shaded sensor-fire indicators to an axis
def add_sensor_indicators(ax):
    ymin, ymax = ax.get_ylim()
    for k in k_s1:
        ax.axvline(k, color='steelblue', alpha=0.15, linewidth=6, zorder=0)
    for k in k_s2:
        ax.axvline(k, color='darkorange', alpha=0.15, linewidth=6, zorder=0)
 
# --- x1 ---
ax = axes[0]
ax.plot(k_values, x[:, 0], 'k', linewidth=1.4, label="true $x_1$")
ax.plot(k_values, xhat[:, 0], 'r--', linewidth=1.4, label="estimated $\\hat{x}_1$")
ax.set_ylabel("$x_1$")
ax.legend(loc='upper right', fontsize=8)
ax.grid(True, alpha=0.3)
add_sensor_indicators(ax)
 
# --- x2 ---
ax = axes[1]
ax.plot(k_values, x[:, 1], 'k', linewidth=1.4, label="true $x_2$")
ax.plot(k_values, xhat[:, 1], 'r--', linewidth=1.4, label="estimated $\\hat{x}_2$")
ax.set_ylabel("$x_2$")
ax.legend(loc='upper right', fontsize=8)
ax.grid(True, alpha=0.3)
add_sensor_indicators(ax)
 
# --- y1 ---
ax = axes[2]
ax.plot(k_values, y_true_without_noise[:, 0], 'k', linewidth=1.4, label="true $y_1$ (no noise)")
ax.plot(k_values, yhat[:, 0], 'r--', linewidth=1.4, label="estimated $\\hat{y}_1 = 0.5\\hat{x}_1 + 0.5\\hat{x}_2$")
ax.scatter(k_s1, y[k_s1, 0], s=25, color='steelblue', zorder=5, label="sensor 1 measurement ($k$=0,5,10,...)")
ax.scatter(k_s2, y[k_s2, 1], s=25, color='darkorange', zorder=5, label="sensor 2 measurement ($k$=1,6,11,...)")
ax.set_ylabel("$y_1$")
ax.legend(loc='upper right', fontsize=8)
ax.grid(True, alpha=0.3)
add_sensor_indicators(ax)
 
# --- y2 ---
ax = axes[3]
ax.plot(k_values, y_true_without_noise[:, 1], 'k', linewidth=1.4, label="true $y_2$ (no noise)")
ax.plot(k_values, yhat[:, 1], 'r--', linewidth=1.4, label="estimated $\\hat{y}_2 = \\hat{x}_2$")
ax.scatter(k_s1, y[k_s1, 0], s=25, color='steelblue', zorder=5, label="sensor 1 measurement ($k$=0,5,10,...)")
ax.scatter(k_s2, y[k_s2, 1], s=25, color='darkorange', zorder=5, label="sensor 2 measurement ($k$=1,6,11,...)")
ax.set_ylabel("$y_2$")
ax.set_xlabel("$k$")
ax.legend(loc='upper right', fontsize=8)
ax.grid(True, alpha=0.3)
add_sensor_indicators(ax)
 
# shared legend for shading
s1_patch = mpatches.Patch(color='steelblue', alpha=0.3, label='sensor 1 fires')
s2_patch = mpatches.Patch(color='darkorange', alpha=0.3, label='sensor 2 fires')
fig.legend(handles=[s1_patch, s2_patch], loc='lower center', ncol=2,
           fontsize=9, bbox_to_anchor=(0.5, 0.0))
 
plt.tight_layout(rect=[0, 0.03, 1, 1])
os.makedirs("results", exist_ok=True)
plt.savefig("results/plot1_states.png", dpi=150, bbox_inches='tight')


# -----------------------------
# Plot Kalman gain elements
# -----------------------------
plt.figure(figsize=(10, 6))

plt.plot(k_values, K_store[:, 0, 0], "o-", label="K11, sensor 1 to x1")
plt.plot(k_values, K_store[:, 1, 0], "o-", label="K21, sensor 1 to x2")
plt.plot(k_values, K_store[:, 0, 1], "o-", label="K12, sensor 2 to x1")
plt.plot(k_values, K_store[:, 1, 1], "o-", label="K22, sensor 2 to x2")

plt.xlabel("k")
plt.ylabel("Kalman gain")
plt.title("Kalman gain elements")
plt.legend()
plt.grid(True)

plt.tight_layout()


# ================================================================
# Plot 2: Covariance with sensor indicators and physical labels
# ================================================================
fig, ax = plt.subplots(figsize=(13, 5))
 
ax.plot(k_values, P_store[:, 0, 0], color='steelblue',   linewidth=1.8,
        label="$P_{11}$ — variance of $x_1$ estimate")
ax.plot(k_values, P_store[:, 1, 1], color='darkorange',  linewidth=1.8,
        label="$P_{22}$ — variance of $x_2$ estimate")
ax.plot(k_values, P_store[:, 0, 1], color='seagreen',    linewidth=1.4, linestyle='--',
        label="$P_{12}=P_{21}$ — cross-covariance ($x_1$–$x_2$ correlation)")
 
# shaded bands for sensor fires
ymin, ymax = ax.get_ylim()
for k in k_s1:
    ax.axvline(k, color='steelblue', alpha=0.12, linewidth=7, zorder=0)
for k in k_s2:
    ax.axvline(k, color='darkorange', alpha=0.12, linewidth=7, zorder=0)
 
ax.set_xlabel("$k$")
ax.set_ylabel("Covariance value")
ax.set_title("Covariance matrix $P$ — periodic rise (predict) and fall (measurement update)")
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3)
 
s1_patch = mpatches.Patch(color='steelblue', alpha=0.3, label='sensor 1 fires')
s2_patch = mpatches.Patch(color='darkorange', alpha=0.3, label='sensor 2 fires')
ax.legend(handles=ax.get_legend_handles_labels()[0] + [s1_patch, s2_patch],
          labels=ax.get_legend_handles_labels()[1] + ['sensor 1 fires', 'sensor 2 fires'],
          loc='upper right', fontsize=8)
 
plt.tight_layout()
plt.savefig("results/plot2_covariance.png", dpi=150, bbox_inches='tight')
plt.show()