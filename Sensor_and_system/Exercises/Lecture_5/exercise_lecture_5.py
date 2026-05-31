import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Exercise 1 - 1D Kalman Filter for IMU (perfect model/estimator match)
# Replicate the four-panel figure from slide 25.
#
# One accelerometer measures:  y_k = a_k + b_k + noise
# Four states estimated:       x (position), v (velocity),
#                               a (acceleration), b (bias)
#
# Only the accelerometer is used - no gyroscope or magnetometer.
# Because the accelerometer is the ONLY sensor, x and v are unobservable
# (not in the measurement), so they drift even with a perfect filter.
# The filter CAN recover a and b from the single noisy measurement.
# =============================================================================

# ── Parameters ────────────────────────────────────────────────────────────────
Ts    = 0.001   # sample period (s) - 1 kHz IMU
phi   = 0.9     # AR(1) coefficient: how correlated a_{k+1} is to a_k
                # phi<1 means mean-reverting; phi=0 pure white noise

sigma_wa = 0.1  # std of random kick to acceleration each step (process noise)
sigma_wb = 0    # std of random kick to bias each step
                # sigma_wb = 0 → perfect model match: true bias is constant
                # and filter models it as constant → K^[b] → 0 → bias locks in
sigma_v  = 0.1  # std of accelerometer measurement noise

b_true = -0.2   # true constant bias the filter must discover

N = 50_000      # number of steps (50 s at 1 kHz)

# ── System matrices ───────────────────────────────────────────────────────────
# State vector: chi_k = [x, v, a, b]^T
# Row 1: x_{k+1} = x_k + Ts*v_k          (Euler position)
# Row 2: v_{k+1} = v_k + Ts*a_k          (Euler velocity)
# Row 3: a_{k+1} = phi*a_k + noise        (AR(1) acceleration)
# Row 4: b_{k+1} = b_k + noise            (bias random walk, 0 here)
Phi = np.array([
    [1, Ts,  0,   0],
    [0,  1,  Ts,  0],
    [0,  0,  phi, 0],
    [0,  0,  0,   1]
])

# Process noise covariance - only a and b receive random kicks
Q = np.diag([0, 0, sigma_wa**2, sigma_wb**2])

# Measurement matrix - accelerometer reads a + b, not x or v
H = np.array([[0, 0, 1, 1]])

# Measurement noise variance
R = np.array([[sigma_v**2]])

# ── Storage ───────────────────────────────────────────────────────────────────
# Rows: [x, v, a, b] - one column per time step
state_true = np.zeros((4, N))   # true states
state_hat  = np.zeros((4, N))   # KF estimates
P          = np.eye(4)           # initial covariance: equal uncertainty, no correlations

# Initial conditions: all states zero except true bias starts at b_true
state_true[3, 0] = b_true

np.random.seed(42)

# ── Simulation + Kalman filter (single loop) ──────────────────────────────────
for k in range(N - 1):

    # ── True system propagation ───────────────────────────────────────────────
    # Draw 4 independent standard-normal samples, scale the ones that matter:
    # w[2] drives acceleration noise, w[3] drives bias noise (0 here)
    w = np.random.randn(4)
    w_scaled = np.array([0, 0, sigma_wa * w[2], sigma_wb * w[3]])

    state_true[:, k+1] = Phi @ state_true[:, k] + w_scaled

    # Measurement at step k: sensor reads a_k + b_k + noise
    v = sigma_v * np.random.randn()
    y = H @ state_true[:, k] + v

    # ── Kalman filter ─────────────────────────────────────────────────────────

    # Step B: predict next state and covariance from previous estimate
    state_pred = Phi @ state_hat[:, k]
    P_pred     = Phi @ P @ Phi.T + Q     # P grows here (uncertainty increases)

    # Step A: measurement update - correct prediction with new measurement
    S          = H @ P_pred @ H.T + R                            # innovation covariance
    K          = P_pred @ H.T @ np.linalg.inv(S)                 # Kalman gain
    innovation = y - H @ state_pred                              # surprise: actual - predicted
    state_hat[:, k+1] = state_pred + (K @ innovation).flatten()  # corrected estimate
    P = (np.eye(4) - K @ H) @ P_pred                            # P shrinks here (uncertainty decreases)

# ── Plot ──────────────────────────────────────────────────────────────────────
time = np.arange(N) * Ts

fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)

labels = [
    (state_true[0], state_hat[0], r'$x_k, \hat{x}_k$',  False),
    (state_true[1], state_hat[1], r'$v_k, \hat{v}_k$',  False),
    (state_true[2], state_hat[2], r'$a_k, \hat{a}_k$',  False),
    (state_true[3], state_hat[3], r'$b_k, \hat{b}_k$',  True),
]

for ax, (true_sig, est_sig, ylabel, dashed) in zip(axes, labels):
    ax.plot(time, true_sig, 'b--' if dashed else 'b-', lw=1.2, label='True')
    ax.plot(time, est_sig,  'r-', lw=0.8, alpha=0.85, label='KF estimate')
    ax.set_ylabel(ylabel, fontsize=9)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel('$t$  (s)', fontsize=10)

# Title reads directly from the parameter variables above
fig.suptitle(
    f'Exercise 1 — 1D IMU Kalman Filter (perfect model match)\n'
    rf'$\phi={phi},\ \sigma_{{w_a}}={sigma_wa},\ \sigma_{{w_b}}={sigma_wb},'
    rf'\ \sigma_{{\nu}}={sigma_v},\ b={b_true},\ T_s={Ts}$',
    fontsize=11
)

plt.tight_layout()
plt.savefig('exercise1_slide25_replication.png', dpi=150, bbox_inches='tight')
plt.show()
