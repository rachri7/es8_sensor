"""
# =============================================================================
# BASIC KALMAN FILTER — Questions (Task 7.3)
# =============================================================================
 
# Q1 — How to set discrete parameters phi and sigma_qa for acceleration?
#
# phi = exp(-omega_b * Ts) is the AR(1) pole — how much the acceleration state
# remembers from the previous step. High omega_b → phi closer to 0 → acceleration
# can change fast (good for aggressive movement). Low omega_b → phi closer to 1
# → acceleration changes slowly (good for gentle sliding).
# sigma_qa is derived from your design choice sigma_a via PDF eq (2b):
#   sigma_qa = sigma_a * sqrt(1 - phi²)
# sigma_a is your physical guess for how large real accelerations will be (~0.5 m/s²
# for gentle hand motion). sigma_qa then sets how much random kick Q gives the
# acceleration state each timestep. They are linked — you do not set them independently.
 
 
# Q2 — How is measurement noise R chosen?
#
# R = std² taken directly from the static recording (sensor_analysis).
# Arduino sitting still → any variation is purely sensor noise, not motion.
# std is the noise standard deviation, R = std² is the variance fed into the KF.
# This is the only parameter taken from measured data — it is not tuned.
 
 
# Q3 — Why is initial state set to zero?
#
# We know the Arduino starts at rest at position x=0 with zero velocity.
# So x0 = [0, 0, 0] is the correct and known initial condition.
# Initial covariance P0 is set small (not zero) to reflect we are confident
# but not perfectly certain about this starting point.
 
 
# Q4 — What do the ±2σ plots show? What do we see?
#
# The ±2σ band is the filter's own estimate of its uncertainty at each timestep.
# For KF3, the velocity and position bands are extremely narrow — the filter
# thinks it knows these states well. But the end drift (v=0.26 m/s, p=0.71 m)
# clearly falls far outside the band → filter is overconfident because bias is
# not modelled. The unmodelled bias integrates silently into velocity and position
# while the filter has no way to account for it in its uncertainty estimate.
 
 
# =============================================================================
# EXTENDED KALMAN FILTER WITH BIAS — Questions (Task 7.4)
# =============================================================================
 
# Q1 — What to use for initial bias variance P0[4,4]?
#
# Set P0[4,4] = R (sensor variance from static recording). This says we expect
# the bias to be roughly on the order of the sensor noise std. We do not know
# the exact bias at start (it varies between Arduinos), so this is a reasonable
# starting scale. The filter will converge to the true value regardless.
 
 
# Q2 — Any reason to choose R differently when bias state is included?
#
# In principle no — R stays the same value from the static recording.
# R represents only the white noise part of the sensor. The bias is now a
# separate state so R does not need to absorb it anymore.
# You could split the static variance into noise + bias contributions if you
# had prior knowledge of each, but only if you are confident in that split —
# making R smaller than the true noise makes the filter overconfident in
# measurements. Since our sensor noise is already very small, keeping R = std²
# unchanged is the safe and correct choice.
 
 
# Q3 — What do the ±2σ plots show for KF4? What changes?
#
# The KF4 ±2σ bands for velocity and position are wide (v: ±1.95, p: ±4.14)
# meaning the filter honestly admits large uncertainty about these states.
# The true end condition (zero) falls inside or very close to this band —
# so KF4 is correctly calibrated even though estimates still drift somewhat.
# The bias estimate converges toward the measured static bias (green line),
# confirming the bias state is working.
# The remaining drift comes from position being fundamentally unobservable
# without a position sensor — the KF can only do so much with acceleration alone.
#
# To improve convergence: increase sigma_qb slightly if bias settles too slowly.
# The bias random walk model (eq 4a) is appropriate for a constant or slowly
# drifting bias — no reason to change it for this exercise.
 
 
# Q4 — Does it matter if P0[1:3,1:3] = I instead of zeros?
#
# Yes. Setting P0 = I for a, v, p means we start with larger uncertainty on
# all states, so the filter initially trusts measurements more heavily and
# takes longer to settle. The ±2σ bands grow larger at the start and the
# end variance is higher because initial uncertainty propagates forward through
# the kinematic integration. For this exercise we know the start conditions
# well (zero velocity, zero position), so keeping P0 small for a, v, p and
# only setting P0[4,4] large for bias is the more honest and correct choice.

Summary For Parameters:
  R         -> measured from data, fixed, take the variance measured from static
  sigma_a   -> you choose based on expected motion, used to derive sigma_qa
  sigma_qb  -> you choose based on how stable you think the bias is

Usage
-----
  python kalman_ins.py --static static.csv --motion motion.csv
  python kalman_ins.py --demo                    # synthetic data, no hardware needed
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy import signal as sp_signal

# ── constants ────────────────────────────────────────────────────────────────
G_TO_MS2 = 9.81      # 1 g in m/s²
FS       = 13.0      # sample rate Hz
TS       = 1.0 / FS  # sample period s  (≈ 0.0769 s)


################################################
## Helper function for load or possible demo
###############################################

def load_csv(path: str, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    Load accelerometer CSV with format:  ax, ay, az  (in g, no header, no timestamp)
    e.g.   0.01074219,-0.01831055,1.02587891
 
    axis: which column to use as the measurement axis
          0 = x  (default — direction of sliding on the table)
          1 = y
          2 = z  (vertical, includes ~1g gravity when flat)
 
    Time is reconstructed from sample index assuming constant FS (13 Hz).
    Values are converted from g to m/s².
    """
    data  = np.loadtxt(path, delimiter=",")   # no header to skip
    n     = data.shape[0]
    t_s   = np.arange(n) * TS                 # reconstruct time from sample index
    a_ms2 = data[:, axis] * G_TO_MS2          # selected axis, g -> m/s²
    return t_s, a_ms2


# TO Create fake data Probally remove
def demo_data(duration_static=60.0, duration_motion=25.0, seed=42):
    """
    Synthetic data designed to clearly show the difference between all three methods.

    Key design choices:
      - Motion is 25 s so the 4-state KF has time to converge the bias estimate
      - True acceleration is a realistic windowed sinusoid — starts and ends at
        zero velocity and zero net displacement (like a careful hand movement)
      - Bias is 0.08 g — large enough to cause obvious drift when not corrected
      - Sensor noise is 0.015 g — small relative to bias, so bias dominates
        the drift problem, which is exactly the point of the exercise

    Expected results:
      benchmark  -> large position drift   (bias integrates unboundedly)
      3-state KF -> moderate improvement   (bias not modelled, still drifts)
      4-state KF -> close to zero at end   (bias estimated and removed)
    """
    rng = np.random.default_rng(seed)

    # Bias >> noise: this is the realistic Arduino case and the core problem
    true_bias    = 0.08 * G_TO_MS2    # 0.08 g — realistic Arduino offset
    sensor_noise = 0.015 * G_TO_MS2   # 0.015 g — typical IMU noise floor

    # static recording
    n_s      = int(duration_static * FS)
    t_static = np.arange(n_s) * TS
    a_static = true_bias + rng.normal(0, sensor_noise, n_s)

    # motion recording
    # True acceleration: windowed sinusoid — tapers to zero at both ends so
    # true velocity and position naturally return to zero at the end.
    # This mimics two gentle forth-and-back slides on the table.
    n_m      = int(duration_motion * FS)
    t_motion = np.arange(n_m) * TS
    freq     = 2.0 / duration_motion                         # 2 cycles total
    window   = np.sin(np.pi * t_motion / duration_motion)   # tapers start+end
    a_true   = 0.25 * window * np.sin(2 * np.pi * freq * t_motion)

    # True velocity and position (ground truth — only exists in demo mode)
    v_true = np.cumsum(a_true) * TS
    p_true = np.cumsum(v_true) * TS

    # Measured signal = true acceleration + constant bias + white noise
    a_motion = a_true + true_bias + rng.normal(0, sensor_noise, n_m)

    print(f"  [demo] True end velocity : {v_true[-1]:.4f} m/s  (should be ~0)")
    print(f"  [demo] True end position : {p_true[-1]:.4f} m   (should be ~0)")
    print(f"  [demo] True bias         : {true_bias:.4f} m/s²  ({true_bias/G_TO_MS2:.3f} g)")
    print(f"  [demo] Sensor noise std  : {sensor_noise:.4f} m/s²  ({sensor_noise/G_TO_MS2:.3f} g)")
    print(f"  [demo] Bias/noise ratio  : {true_bias/sensor_noise:.1f}x  (bias >> noise)")

    return t_static, a_static, t_motion, a_motion, true_bias, v_true, p_true


# ---- STEP 1 — SENSOR ANALYSIS: Measure Bais and Process Noise std ----

def sensor_analysis(t: np.ndarray, a: np.ndarray) -> dict:
    """
    Compute bias and noise statistics from a static (stationary) recording.

    ASSUMPTION: Arduino was perfectly still during this recording.
    Therefore any variation in the signal is purely sensor noise, not motion.
    This gives us:
      - bias = mean(a)        the constant offset the sensor adds
      - std  = std(a)         the noise standard deviation
      - R    = std²           the measurement noise variance for the KF
    """
    bias = np.mean(a)
    std  = np.std(a, ddof=1)
    var  = std ** 2

    result = dict(bias=bias, std=std, var=var, min=a.min(), max=a.max(), n=len(a))

    print("=" * 60)
    print("STEP 1 — SENSOR ANALYSIS")
    print("=" * 60)
    print(f"  Samples        : {len(a)}  ({len(a)/FS:.1f} s at {FS} Hz)")
    print(f"  Bias (mean)    : {bias:+.5f} m/s²  =  {bias/G_TO_MS2:+.5f} g")
    print(f"  Noise std      : {std:.5f} m/s²  =  {std/G_TO_MS2:.5f} g")
    print(f"  R = std²       : {var:.6f} m²/s⁴   <- this goes into the KF")
    print(f"  Min / Max      : {a.min():.5f} / {a.max():.5f} m/s²")
    print()
    print("  R is the ONLY parameter taken from measured data.")
    print("  sigma_a and sigma_qb are design choices, set separately.")

    fig, axes = plt.subplots(1, 1, figsize=(10, 5), tight_layout=True)
    fig.suptitle("Step 1 — Sensor analysis (static recording)", fontweight="bold")

    axes.plot(t, a, lw=0.6, color="#1a6eb5", label="raw accel")
    axes.axhline(bias,         color="#d84c30", lw=1.5, label=f"bias = {bias:.4f} m/s²")
    axes.axhline(bias + 2*std, color="#d84c30", lw=0.8, ls="--", label="-+2σ")
    axes.axhline(bias - 2*std, color="#d84c30", lw=0.8, ls="--")
    axes.set_ylabel("Acceleration (m/s²)")
    axes.set_xlabel("Time (s)")
    axes.legend(fontsize=9)
    axes.grid(True, alpha=0.3)
    axes.set_title(f"R = std² = {var:.6f} m²/s⁴  (measurement noise variance)")

    plt.savefig("sensor_analysis.png", dpi=150)
    plt.show()
    return result

# ---- STEP 2 — BENCHMARK: direct double integration (pure kinematics)

# Can Remove just for your own NOTES
'''
    EXPECTED RESULT: velocity and position drift badly.
    Any small sensor bias accumulates unboundedly through integration.
    This is WHY the Kalman filter is needed.
'''
def benchmark(t: np.ndarray, a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Integrate acceleration once for velocity, twice for position.
    ASSUMPTION: v(0) = 0, p(0) = 0  (known start condition).
    """
    n = len(a)
    v = np.zeros(n)
    p = np.zeros(n)
    for k in range(n - 1):
        v[k+1] = v[k] + a[k] * TS
        p[k+1] = p[k] + v[k] * TS
    return v, p


# KALMAN FILTER — ALGORITHM RUN (Same for both just different H and PHI Matrixs)
def run_kf(measurements: np.ndarray, Phi, Q, H, R, x0, P0):
    """
    Standard linear Kalman filter.
    Returns:
      x_hist  (n, nx)  — state estimates at each timestep
      std_hist (n, nx) — std dev of each state (sqrt of diagonal of P)
                         used for plotting +-2 sigma confidence intervals
    """
    n  = len(measurements)
    nx = len(x0)
    x  = x0.copy().astype(float)
    P  = P0.copy().astype(float)

    x_hist   = np.zeros((n, nx))
    std_hist = np.zeros((n, nx))

    for k in range(n):
        # --- measurement update ---
        
        innov = np.array([[measurements[k]]]) - H @ x.reshape(-1, 1)    # Innovation: measured - predicted
        S = H @ P @ H.T + R                                             # total uncertianty
        K = P @ H.T @ np.linalg.inv(S)                                  # Kalman gain: (model uncertianty/ total uncertianty)
        
        # Update state and covariance by kalman gain scaling
        x = x + (K @ innov).flatten()
        P = (np.eye(nx) - K @ H) @ P

    
        x_hist[k]   = x                         # state estimates
        std_hist[k] = np.sqrt(np.diag(P))       # std dev of each state (for +-2 sigma confidence intervals plot)

        # ---- time update (predict next state) forward throught state transtion matrix ---
        x = Phi @ x
        P = Phi @ P @ Phi.T + Q

    return x_hist, std_hist

###################################################
### REMOVE COMMENT FOR KALMAN 3a,3b ON YOUR OWN
#################################################

# ---- STEP 3a — 3-STATE KALMAN FILTER (no bias) - States: x = [a, v, p] ----

def kf_no_bias(t, a, sensor_stats, omega_b, sigma_a):
    """
    3-state KF following eq (5)-(6) in PDF, without bias state.

    State transition matrix Phi (forward Euler, eq 3):
      a(k+1) = phi * a(k) + w_a     ← AR(1) model for acceleration
      v(k+1) = v(k) + a(k)*Ts       ← kinematics
      p(k+1) = p(k) + v(k)*Ts       ← kinematics

    Measurement model H = [1, 0, 0]:
      y(k) = a(k) + v_k             ← we only measure acceleration

    PARAMETER CHOICES:
      phi     : AR(1) pole = exp(-omega_b * Ts)
                Controls how fast acceleration state changes.
                omega_b is the bandwidth you choose for expected hand motion.

      sigma_qa: process noise on acceleration state, derived from PDF eq (2b):
                sigma²_qa = sigma²_a * (1 - phi²)
                sigma_a is YOUR choice of expected motion magnitude.
                This has NOTHING to do with R (sensor noise).

      R       : measurement noise variance, from static recording.
                This is FIXED — comes from sensor_analysis().

      Q       : process noise matrix. Only acceleration has process noise.
                Velocity and position are derived by kinematics so Q[1,1]=Q[2,2]=0.
                ASSUMPTION: kinematic equations are exact given true acceleration.

      P0      : initial covariance. We know start conditions well (v=0, p=0)
                so we set P0 small. Small ≠ zero — zero would mean perfect
                certainty which is unrealistic.
    """
    # phi from eq (2a): AR(1) pole for acceleration model
    phi = np.exp(-omega_b * TS)

    # sigma_qa from PDF eq (2b) rearranged:
    # sigma²_a = sigma²_qa / (1 - phi²)  ->  sigma_qa = sigma_a * sqrt(1-phi²)
    # ASSUMPTION: sigma_a represents the typical magnitude of hand acceleration.
    # It is a DESIGN parameter, not measured. Typical range: 0.2 – 1.0 m/s².
    sigma_qa = sigma_a * np.sqrt(1 - phi**2)

    Phi = np.array([
        [phi, 0,  0],
        [TS,  1,  0],
        [0,  TS,  1],
    ])

    # Q: only acceleration has process noise (eq 5b from PDF)
    Q = np.diag([sigma_qa**2, 0.0, 0.0])

    # H: we measure acceleration only (eq 7 from PDF, without bias)
    H = np.array([[1.0, 0.0, 0.0]])

    # R: from static sensor measurement — NOT a tuning parameter
    R = np.array([[sensor_stats["var"]]])

    # Initial state: known to be zero (start at rest)
    x0 = np.zeros(3)

    # Initial covariance: small diagonal
    # ASSUMPTION: we are fairly confident about initial conditions.
    # Using sensor variance as a rough scale for initial uncertainty.
    P0 = np.diag([sensor_stats["var"], 1e-6, 1e-6])


    x_est, std_est = run_kf(a, Phi, Q, H, R, x0, P0)

    return x_est, std_est


# --- STEP 3b — 4-STATE KALMAN FILTER (with bias estimation) - x = [a, v, p, b] ----

def kf_with_bias(t, a, sensor_stats, omega_b, sigma_a, sigma_qb):
    """
    4-state KF following eq (5b) in PDF, with bias state added.

    The bias state b models the slowly-varying sensor offset.
    Bias model (eq 4a): b(k+1) = b(k) + w_b   (random walk)
    Measurement model (eq 6): y(k) = a(k) + b(k) + v_k
    So H = [1, 0, 0, 1] — measurement depends on both acceleration and bias.

    WHY ADD BIAS STATE:
    Without it, a constant sensor offset is mistaken for real acceleration,
    which integrates into velocity and position drift (seen in benchmark).
    The 4-state KF estimates and removes this offset automatically.

    PARAMETER CHOICES vs 3-STATE KF:
      phi, sigma_qa, R : same reasoning as 3-state KF above.

      sigma_qb : process noise on bias random walk.
                 ASSUMPTION: bias is nearly constant but can drift slowly.
                 Small sigma_qb -> bias changes very slowly (conservative).
                 Large sigma_qb -> bias can change quickly (more adaptive).
                 Typical starting value: much smaller than sigma_qa.
                 Rule of thumb: sigma_qb << sigma_qa

      R: same as before — from static recording.
         ASSUMPTION: since we now model bias separately as a state,
         R should represent ONLY the white noise part of the sensor.
         The bias (systematic offset) is now handled by the bias state,
         not lumped into R. So R stays the same value from step 1.

      P0[3,3] (initial bias covariance):
         ASSUMPTION: we don't know the initial bias exactly (it varies
         between Arduinos). We set P0[3,3] = sensor variance as a starting
         guess — it says "I expect the bias to be roughly on the order of
         the sensor noise std." The filter will converge to the true value.
         Using the measured bias from static data would be better if
         you're confident it won't change, but the PDF says to start at 0.
    """
    phi      = np.exp(-omega_b * TS)
    sigma_qa = sigma_a * np.sqrt(1 - phi**2)

    # State transition: same as 3-state but with bias appended
    # bias walks as b(k+1) = b(k) + w_b  -> Phi[3,3] = 1
    Phi = np.array([
        [phi, 0,  0, 0],
        [TS,  1,  0, 0],
        [0,  TS,  1, 0],
        [0,   0,  0, 1],
    ])

    # Q: process noise on acceleration AND bias
    # Velocity and position still have no direct process noise (kinematics)
    Q = np.diag([sigma_qa**2, 0.0, 0.0, sigma_qb**2])

    # H: measurement = acceleration state + bias state (eq 6 from PDF)
    H = np.array([[1.0, 0.0, 0.0, 1.0]])

    # R: same as 3-state — from static recording
    R = np.array([[sensor_stats["var"]]])

    # Initial state: all zero (PDF says x0 = 0, b0 = 0)
    x0 = np.zeros(4)

    # Initial covariance:
    # a,v,p (acceleration,velocity,postion) all set to zero we know start at this points

    # Set (bias variance) P0[4,4] = R (sensor variance from static recording). This says we expect
    # the bias to be roughly on the order of the sensor noise std. We do not know
    # the exact bias at start (it varies between Arduinos), so this is a reasonable
    # starting scale. The filter will converge to the true value regardless

    P0 = np.diag([0, 0, 0, sensor_stats["var"]])
    
    # Reflection quistion(4) set to Identity, this means we unsure about a,v,p but we know its zero at start
    #P0 = np.diag([1, 1, 1, sensor_stats["var"]])


    print("\n Variables set for the Kalman Filter (with bias estimation)")
    print(f"  omega_b   = {omega_b:.3f} rad/s   ->  phi = {phi:.5f}")
    print(f"  sigma_a   = {sigma_a:.4f} m/s²  (design choice, same as 3-state KF)")
    print(f"  sigma_qa  = {sigma_qa:.5f} m/s²  (derived from sigma_a)")
    print(f"  sigma_qb  = {sigma_qb:.5f} m/s²  (design choice: how fast bias drifts)")
    print(f"  Q[0,0]    = {sigma_qa**2:.6f}  Q[3,3] = {sigma_qb**2:.8f}  m²/s⁴  (Process noise variance for Acceleration and Bais)")  
    print(f"  R         = {sensor_stats['var']:.6f} m²/s⁴  (from static recording — fixed)")
    print(f"  P0[3,3]   = {sensor_stats['var']:.6f}  (initial bias uncertainty)")
    print("\n Note: R and Q[0,0] are independent. Do not confuse them.")

    x_est, std_est = run_kf(a, Phi, Q, H, R, x0, P0)

    print(f"  Converged bias estimate: {x_est[-1, 3]:.5f} m/s²  ({x_est[-1,3]/G_TO_MS2:.5f} g)")

    return x_est, std_est


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def _annotate_final(ax, t, values, color, label):
    """
    Mark the final value of a signal with a dot + text annotation.
    This shows drift clearly — for velocity and position it should be ~0.
    """
    final = values[-1]
    ax.scatter(t[-1], final, color=color, s=30, zorder=5)
    ax.annotate(
        f"end={final:.3f}",
        xy=(t[-1], final),
        xytext=(-48, 8),
        textcoords="offset points",
        fontsize=7,
        color=color,
        arrowprops=dict(arrowstyle="-", color=color, lw=0.6),
    )
 
 
def _mark_zero_conditions(ax, t):
    """
    Mark the known ground-truth zero conditions at start and end.
    We know for certain: start = 0 (placed at rest), end = 0 (returned to start).
    These are the only ground truth reference points we have without a position sensor.
    """
    ax.scatter([t[0],  t[-1]], [0, 0],
               color="green", s=50, zorder=6,
               marker="D", label="known = 0")
 
def plot_results(t, a, v_bench, p_bench, x3, std3, x4, std4, true_bias=None, v_true=None, p_true=None,measured_bias=None):
    """
    Single figure — 2x2 grid, all three methods overlaid on each subplot.
 
    Top-left  : Acceleration — raw sensor + KF3 estimate + KF4 estimate
    Top-right : Velocity     — benchmark + KF3 + KF4 + ±2σ for KF4
    Bottom-left: Position    — benchmark + KF3 + KF4 + ±2σ for KF4
    Bottom-right: Bias       — 4-state KF bias estimate converging over time
 
    Green diamonds = known ground truth zeros (start and end must be zero).
    End-of-run drift annotated on each line so you can compare methods directly.
    In demo mode, true velocity/position shown as green dashed line.
    """
    col_bench = "#888888"
    col_raw   = "#cccccc"
    col_kf3   = "#1a6eb5"
    col_kf4   = "#d84c30"
    alpha_ci  = 0.15
 
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), tight_layout=True)
    fig.suptitle(
        "Kalman filter results — all methods overlaid\n"
        "Green diamonds = known zeros (start and end).  Closer to zero = better.",
        fontweight="bold", fontsize=11
    )
    ax_a, ax_v, ax_p, ax_b = axes[0,0], axes[0,1], axes[1,0], axes[1,1]
 
    # ── top-left: acceleration ────────────────────────────────────────────────
    # Raw sensor + both KF acceleration estimates overlaid.
    # KF3 estimate ≈ a_true + bias  (cannot separate them, no bias state)
    # KF4 estimate ≈ a_true         (bias removed into bias state)
    ax_a.plot(t, a,          lw=2, color="black",ls=":",   label="raw sensor")
    ax_a.plot(t, x3[:, 0],   lw=1.2, color=col_kf3,   label="KF3 â (includes bias)")
    ax_a.fill_between(t, x3[:,0]-2*std3[:,0], x3[:,0]+2*std3[:,0], alpha=alpha_ci, color=col_kf3, label=f"KF3 -+2σ = {2*std3[-1,0]:.4f} in the end")
    ax_a.plot(t, x4[:, 0],   lw=1.2, color=col_kf4,   label="KF4 â (bias removed)")
    ax_a.fill_between(t, x4[:,0]-2*std4[:,0], x4[:,0]+2*std4[:,0], alpha=alpha_ci, color=col_kf4, label=f"KF4 -+2σ = {2*std4[-1,0]:.4f} in the end")
    ax_a.axhline(0, color="k", lw=0.4, ls="--", alpha=0.4)
    ax_a.set_title("Acceleration", fontsize=10)
    ax_a.set_ylabel("m/s²")
    ax_a.set_xlabel("Time (s)")
    ax_a.legend(fontsize=8)
    ax_a.grid(True, alpha=0.25)
 
    # ── top-right: velocity ───────────────────────────────────────────────────
    ax_v.plot(t, v_bench,  lw=2.0, color="black",   ls=":",  label="benchmark", zorder=5)
    ax_v.plot(t, x3[:, 1], lw=1.2, color=col_kf3,           label="3-state KF")
    ax_v.fill_between(t, x3[:,1]-2*std3[:,1], x3[:,1]+2*std3[:,1],
                      alpha=alpha_ci, color=col_kf3, label=f"KF3 -+2σ = {2*std3[-1,1]:.4f} in the end")
    ax_v.plot(t, x4[:, 1], lw=1.5, color=col_kf4,           label="4-state KF")
    ax_v.fill_between(t, x4[:,1]-2*std4[:,1], x4[:,1]+2*std4[:,1],
                      alpha=alpha_ci, color=col_kf4, label=f"KF4 -+2σ = {2*std4[-1,1]:.4f} in the end")
    if v_true is not None:
        ax_v.plot(t, v_true, lw=1.2, color="green", ls="-.", label="true (demo)")
    ax_v.axhline(0, color="k", lw=0.5, ls="--", alpha=0.5)
    _mark_zero_conditions(ax_v, t)
    _annotate_final(ax_v, t, v_bench,  col_bench, "bench")
    _annotate_final(ax_v, t, x3[:, 1], col_kf3,   "KF3")
    _annotate_final(ax_v, t, x4[:, 1], col_kf4,   "KF4")
    ax_v.set_title("Velocity", fontsize=10)
    ax_v.set_ylabel("m/s")
    ax_v.set_xlabel("Time (s)")
    ax_v.legend(fontsize=8)
    ax_v.grid(True, alpha=0.25)
 
    # ── bottom-left: position ─────────────────────────────────────────────────
    ax_p.plot(t, p_bench,  lw=2.0, color="black",   ls=":",  label="benchmark", zorder=5)
    ax_p.plot(t, x3[:, 2], lw=1.2, color=col_kf3,           label="3-state KF")
    ax_p.fill_between(t, x3[:,2]-2*std3[:,2], x3[:,2]+2*std3[:,2],
                      alpha=alpha_ci, color=col_kf3, label=f"KF3 -+2σ = {2*std3[-1,2]:.4f} in the end")
    ax_p.plot(t, x4[:, 2], lw=1.5, color=col_kf4,           label="4-state KF")
    ax_p.fill_between(t, x4[:,2]-2*std4[:,2], x4[:,2]+2*std4[:,2],
                      alpha=alpha_ci, color=col_kf4, label=f"KF4 -+2σ = {2*std4[-1,2]:.4f} in the end")
    if p_true is not None:
        ax_p.plot(t, p_true, lw=1.2, color="green", ls="-.", label="true (demo)")
    ax_p.axhline(0, color="k", lw=0.5, ls="--", alpha=0.5)
    _mark_zero_conditions(ax_p, t)
    _annotate_final(ax_p, t, p_bench,  col_bench, "bench")
    _annotate_final(ax_p, t, x3[:, 2], col_kf3,   "KF3")
    _annotate_final(ax_p, t, x4[:, 2], col_kf4,   "KF4")
    ax_p.set_title("Position", fontsize=10)
    ax_p.set_ylabel("m")
    ax_p.set_xlabel("Time (s)")
    ax_p.legend(fontsize=8)
    ax_p.grid(True, alpha=0.25)
 
    # ── bottom-right: bias estimate ───────────────────────────────────────────
    ax_b.plot(t, x4[:, 3], lw=1.5, color=col_kf4, label="KF4 bias estimate b̂")
    ax_b.fill_between(t, x4[:,3]-2*std4[:,3], x4[:,3]+2*std4[:,3],
                      alpha=0.2, color=col_kf4, label=f"KF4 -+2σ = {2*std4[-1,3]:.4f} in the end")
    if true_bias is not None:
        ax_b.axhline(true_bias, color="k", lw=1.2, ls="--",
                     label=f"true bias = {true_bias:.4f} m/s²")
    if measured_bias is not None:
        ax_b.axhline(measured_bias, color="green", lw=1.5, ls="-",
                     label=f"measured bias (static) = {measured_bias:.4f} m/s²")
    ax_b.axhline(0, color="k", lw=0.4, ls=":", alpha=0.4)
    _annotate_final(ax_b, t, x4[:, 3], col_kf4, "converged")
    ax_b.set_title("Bias estimate (4-state KF only)", fontsize=10)
    ax_b.set_ylabel("m/s²")
    ax_b.set_xlabel("Time (s)")
    ax_b.legend(fontsize=8)
    ax_b.grid(True, alpha=0.25)
 
    fig.savefig("kf_results.png", dpi=150)
    plt.show()
 
    # ── bottom-left: position ─────────────────────────────────────────────────
    ax_p.plot(t, p_bench,  lw=2.0, color="black",   ls=":",  label="benchmark", zorder=5)
    ax_p.plot(t, x3[:, 2], lw=1.2, color=col_kf3,           label="3-state KF")
    ax_p.fill_between(t, x3[:,2]-2*std3[:,2], x3[:,2]+2*std3[:,2],
                      alpha=alpha_ci, color=col_kf3, label="KF3 ±2σ")
    ax_p.plot(t, x4[:, 2], lw=1.5, color=col_kf4,           label="4-state KF")
    ax_p.fill_between(t, x4[:,2]-2*std4[:,2], x4[:,2]+2*std4[:,2],
                      alpha=alpha_ci, color=col_kf4, label="KF4 ±2σ")
    if p_true is not None:
        ax_p.plot(t, p_true, lw=1.2, color="green", ls="-.", label="true (demo)")
    ax_p.axhline(0, color="k", lw=0.5, ls="--", alpha=0.5)
    _mark_zero_conditions(ax_p, t)
    _annotate_final(ax_p, t, p_bench,  col_bench, "bench")
    _annotate_final(ax_p, t, x3[:, 2], col_kf3,   "KF3")
    _annotate_final(ax_p, t, x4[:, 2], col_kf4,   "KF4")
    ax_p.set_title("Position", fontsize=10)
    ax_p.set_ylabel("m")
    ax_p.set_xlabel("Time (s)")
    ax_p.legend(fontsize=8)
    ax_p.grid(True, alpha=0.25)
 
    # ── bottom-right: bias estimate ───────────────────────────────────────────
    ax_b.plot(t, x4[:, 3], lw=1.5, color=col_kf4, label="KF4 bias estimate b̂")
    ax_b.fill_between(t, x4[:,3]-2*std4[:,3], x4[:,3]+2*std4[:,3],
                      alpha=0.2, color=col_kf4, label="±2σ")
    if true_bias is not None:
        ax_b.axhline(true_bias, color="k", lw=1.2, ls="--",
                     label=f"true bias = {true_bias:.4f} m/s²")
    ax_b.axhline(0, color="k", lw=0.4, ls=":", alpha=0.4)
    _annotate_final(ax_b, t, x4[:, 3], col_kf4, "converged")
    ax_b.set_title("Bias estimate (4-state KF only)", fontsize=10)
    ax_b.set_ylabel("m/s²")
    ax_b.set_xlabel("Time (s)")
    ax_b.legend(fontsize=8)
    ax_b.grid(True, alpha=0.25)
 
    fig.savefig("kf_results.png", dpi=150)
    plt.show()
 

##########################################################
##### ON YOUR OWN PLS ALSO CHANGE THIS PARAMETER GUIDE TO WHAT MAKE SENSE FOR SITUANTION
######################################################

def main():
    parser = argparse.ArgumentParser(
        description="Kalman filter INS/IMU exercise (Python)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
   --omega_b   float   Acceleration bandwidth in rad/s. Controls how fast
                      the acceleration state is allowed to change.
                      phi = exp(-omega_b * Ts). Default: 0.5
                      Try 1.0 or 2.0 if estimates look too sluggish.

  --sigma_a   float   Expected std of true acceleration in m/s².
                      This is your physical guess about motion magnitude.
                      Used to derive sigma_qa via eq (2b) from the PDF.
                      Default: 0.5 m/s²

  --sigma_qb  float   Process noise std on bias random walk (m/s²).
                      Controls how fast the KF allows bias to change.
                      Small -> bias changes slowly (more stable estimate).
                      Large -> bias can track faster changes.
                      Default: 0.002 m/s²
        """)

    parser.add_argument("--static",   type=str,   help="CSV from static recording")
    parser.add_argument("--motion",   type=str,   help="CSV from motion recording")
    parser.add_argument("--demo",     action="store_true", help="Run with synthetic data")
    parser.add_argument("--omega_b",  type=float, default=0.5,   help="Accel bandwidth rad/s (default: 0.5)")
    parser.add_argument("--sigma_a",  type=float, default=0.5,   help="Expected accel std m/s² (default: 0.5)")
    parser.add_argument("--sigma_qb", type=float, default=0.05,  help="Bias random walk std m/s² (default: 0.05)")
    parser.add_argument("--axis",     type=int,   default=0,     help="Accelerometer axis to use: 0=x, 1=y, 2=z (default: 0)")
    args = parser.parse_args()

    # ---- load data -------------
    true_bias = None
    v_true    = None   # only available in demo mode (no ground truth with real hardware)
    p_true    = None

    ##################################################################
    ##### NEED TO REMOVE THIS IF STATEMENT FOR DEMO TOGETHER WITH ARG
    ###################################################################

    if args.demo or (args.static is None and args.motion is None):
        print("Running in DEMO mode with synthetic data.\n")
        t_static, a_static, t_motion, a_motion, true_bias, v_true, p_true = demo_data()
    else:
        if args.static is None or args.motion is None:
            parser.error("Provide both --static and --motion CSV files, or use --demo")
        t_static, a_static = load_csv(args.static, axis=args.axis)
        t_motion, a_motion = load_csv(args.motion, axis=args.axis)

    # ----- step 1: sensor analysis to get R Process noise std and bias ----------------
    sensor_stats = sensor_analysis(t_static, a_static)

    # ----- step 2: benchmark Double intergration -------
    print("=" * 60)
    print("STEP 2 — BENCHMARK (double integration)")
    print("=" * 60)

    v_bench, p_bench = benchmark(t_motion, a_motion)
    print("  Large drift is expected — this shows WHY KF is needed.")

    # ----- step 3a: 3-state KF: Get all state estimations and +-2 sigma -------------------------------
    print("=" * 60)
    print("STEP 3 — Kalman with and without bais state")
    print("=" * 60)

    x3, std3 = kf_no_bias(
        t_motion, a_motion, sensor_stats,
        omega_b=args.omega_b,
        sigma_a=args.sigma_a,
    )

    # ----- step 3b: 4-state KF with bias: Get all state estimations and +-2 sigma ----------------------
    x4, std4 = kf_with_bias(
        t_motion, a_motion, sensor_stats,
        omega_b=args.omega_b,
        sigma_a=args.sigma_a,
        sigma_qb=args.sigma_qb,
    )

    # --------- grading summary -------------------------------
    # For real hardware: judge by how close end position/velocity is to zero.
    t_end = t_motion[-1]
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY — end-of-run drift (should all be ~0 for both as no speed/velocity and same start position)")
    print("=" * 60)
    print(f"  {'Method':<25} {'end pos (m)':>12}  {'end vel (m/s)':>14}")
    print(f"  {'-'*25} {'-'*12}  {'-'*14}")
    print(f"  {'Benchmark':.<25} {p_bench[-1]:>+12.4f}  {v_bench[-1]:>+14.4f}")
    print(f"  {'3-state KF (no bias)':.<25} {x3[-1,2]:>+12.4f}  {x3[-1,1]:>+14.4f}")
    print(f"  {'4-state KF (+ bias)':.<25} {x4[-1,2]:>+12.4f}  {x4[-1,1]:>+14.4f}")
    if true_bias is not None:
        print(f"\n  True bias (demo only) : {true_bias:.4f} m/s²  ({true_bias/G_TO_MS2:.3f} g)")
        print(f"  KF4 bias estimate     : {x4[-1,3]:.4f} m/s²  ({x4[-1,3]/G_TO_MS2:.3f} g)")
        err = abs(x4[-1,3] - true_bias)
        print(f"  Bias estimation error : {err:.4f} m/s²  ({err/true_bias*100:.1f}% of true)")
    # Grade each method by absolute end position error (lower = better)
    errors = {
        "Benchmark":    abs(p_bench[-1]),
        "3-state KF":   abs(x3[-1,2]),
        "4-state KF":   abs(x4[-1,2]),
    }
    # ── plots ─────────────────────────────────────────────────────────────────
    plot_results(t_motion, a_motion, v_bench, p_bench,
                 x3, std3, x4, std4,
                 true_bias=true_bias, v_true=v_true, p_true=p_true,measured_bias=sensor_stats["bias"])


if __name__ == "__main__":
    main()