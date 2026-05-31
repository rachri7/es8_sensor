import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Exercise 2 - Mixed model: 3-state true system vs 4-state Kalman filter
#
# TRUE SYSTEM  (slide 26 model):
#   States [x, v, a] where acceleration is a KNOWN external sinusoidal input.
#   a is not a state to be estimated — it is fed in via Gamma each step.
#
# KALMAN FILTER  (slide 23/25 model):
#   States [x, v, a, b] where the filter still ESTIMATES acceleration
#   as an AR(1) random process with phi=0.9.
#
# This is a MODEL MISMATCH: the true acceleration is deterministic
# (sinusoidal) but the filter models it as stochastic (AR(1)).
# The filter must still separate the sinusoidal a from the constant b
# using only the single noisy measurement y = a_input + b + noise.
# =============================================================================

# ── Parameters ────────────────────────────────────────────────────────────────
Ts      = 0.001   # sample period (s)
phi     = 0.9     # AR(1) coefficient used inside the KF model for acceleration

sigma_wa = 0.1    # KF process noise std for acceleration (AR(1) model)
sigma_wb = 0      # KF process noise std for bias — 0 = constant bias assumed
sigma_v  = 0.1    # measurement noise std

b_true = -0.2     # true constant bias the KF must discover
N      = 50_000   # steps (50 s at 1 kHz)

time = np.arange(N) * Ts

# ── TRUE SYSTEM matrices (3 states: x, v, a_placeholder) ─────────────────────
# Row 1: x_{k+1} = x_k + Ts*v_k
# Row 2: v_{k+1} = v_k + Ts*a_k     (a_k from current state before Gamma reset)
# Row 3: a_{k+1} = 0 + a_input_k    (zeroed by Phi, set by Gamma)
Phi_true = np.array([
    [1, Ts, 0],
    [0,  1, Ts],
    [0,  0,  0]
])

# Gamma sets the a row to the known external input at each step
Gamma = np.array([[0], [0], [1]])

# H_true not used for the true system (no measurement correction on truth)
H_true = np.array([[0, 0, 1]])

# ── KALMAN FILTER matrices (4 states: x, v, a, b) ────────────────────────────
# The KF uses the same 4-state AR(1) model from Exercise 1.
# It does NOT know a is sinusoidal — it estimates it as AR(1) noise.
# Row 3: a_{k+1} = phi*a_k + noise  (filter's internal AR(1) model for a)
# Row 4: b_{k+1} = b_k + noise      (bias random walk, sigma_wb=0 here)
Phi_kf = np.array([
    [1, Ts,  0,   0],
    [0,  1,  Ts,  0],
    [0,  0,  phi, 0],
    [0,  0,  0,   1]
])

# Process noise — only a and b rows receive noise kicks in the KF model
Q    = np.diag([0, 0, sigma_wa**2, sigma_wb**2])
H_kf = np.array([[0, 0, 1, 1]])   # KF measurement: picks a_hat + b_hat
R    = np.array([[sigma_v**2]])

# ── Storage ───────────────────────────────────────────────────────────────────
x_true = np.zeros((3, N))   # true states [x, v, a]
x_hat  = np.zeros((4, N))   # KF estimates [x_hat, v_hat, a_hat, b_hat]
P      = np.eye(4)           # initial covariance — equal uncertainty, no correlations

# True bias is a fixed constant (not a state in the true system)
b = b_true

# Two-frequency sinusoidal acceleration: sum of slow + fast component
# Gives a realistic signal with both low- and high-frequency dynamics
w1      = 2 * np.pi * 0.1    # rad/s — slow component (0.1 Hz, period 10 s)
w2      = 2 * np.pi * 1.0    # rad/s — fast component  (1.0 Hz, period  1 s)
a_input = np.sin(w1 * time) + 0.7 * np.sin(w2 * time)

np.random.seed(42)

# ── Simulation + Kalman filter (single loop) ──────────────────────────────────
for k in range(N - 1):

    # ── True system: propagate with known sinusoidal input ────────────────────
    x_true[:, k+1] = Phi_true @ x_true[:, k] + Gamma.flatten() * a_input[k]

    # Measurement: accelerometer reads true a + true b + noise
    # The KF does not know a — it sees only this combined noisy signal
    y = a_input[k] + b + sigma_v * np.random.randn()

    # ── Kalman filter: 4-state AR(1) estimator ────────────────────────────────

    # Step B: predict using AR(1) model (KF does not use a_input here)
    x_pred = Phi_kf @ x_hat[:, k]
    P_pred = Phi_kf @ P @ Phi_kf.T + Q   # P grows by Q each step

    # Step A: measurement update
    S          = H_kf @ P_pred @ H_kf.T + R
    K          = P_pred @ H_kf.T @ np.linalg.inv(S)
    innovation = y - H_kf @ x_pred        # actual - predicted (a+b)
    x_hat[:, k+1] = x_pred + (K @ innovation).flatten()
    P = (np.eye(4) - K @ H_kf) @ P_pred   # P shrinks after update

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
fig.suptitle(
    f'Exercise 2 — 3-state true system vs 4-state KF (model mismatch)\n'
    rf'$\phi={phi},\ \sigma_{{w_a}}={sigma_wa},\ \sigma_{{w_b}}={sigma_wb},'
    rf'\ \sigma_{{\nu}}={sigma_v},\ b={b_true},\ T_s={Ts}$',
    fontsize=11
)

plot_config = [
    (x_true[0], x_hat[0],                   r'$x_k, \hat{x}_k$', False, 'Position'),
    (x_true[1], x_hat[1],                   r'$v_k, \hat{v}_k$', False, 'Velocity'),
    (a_input,   x_hat[2],                   r'$a_k, \hat{a}_k$', False, 'Acceleration'),
    (np.full(N, b_true), x_hat[3],          r'$b_k, \hat{b}_k$', True,  'Bias'),
]

for ax, (true_sig, est_sig, ylabel, dashed, title) in zip(axes, plot_config):
    ax.plot(time, true_sig, 'b--' if dashed else 'b-', lw=1.2, label='True')
    ax.plot(time, est_sig,  'r-', lw=0.8, alpha=0.85, label='KF estimate')
    ax.set_ylabel(ylabel, fontsize=9)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel('$t$  (s)', fontsize=10)
plt.tight_layout()
plt.savefig('exercise2_slide27_replication.png', dpi=150, bbox_inches='tight')
plt.show()
