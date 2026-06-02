import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2, norm, shapiro
import autograd.numpy as np
from autograd import grad, jacobian
import os


np.random.seed(4)  # makes noise repeatable


# ===============================
# Parameters from exercise 4
# ===============================
a     = 0.95              # AR coefficient, < 1 for stability
k_    = 1                 # desired steady-state gain       ,settles at exactly x* = k*u.
b_p   = k_ * (1 - a)     # input gain: derived from k and it balances how much the new input is weighted against the decay from a

# sin parameters set to 10 more non-linearity
c     = 2                # measurement scaling, analogous to H in linear KF (scales non-linearity term inside the sin)

phi_f = 0.0               # phase offset in process function f (set to 0 initially)
phi_h = 0.0              # phase offset in measurement function h (set to 0 initially)

f_u   = 0.02              # input signal frequency [Hz]

N  = 5000
Ts = 1 / (2 * f_u * 100) # sampling at 100x Nyquist rate for f_u
# (1 / (2 * f_u)) is the nyquist period

time = np.arange(N) * Ts

Q  = np.array([[0.01]])    # process noise variance
R  = np.array([[0.1]])     # measurement noise variance
P0 = np.eye(1) * 1.0       # intial uncertianty coveriance

x0 = np.array([0.0])       # initial state

# noise terms (w, v) are added outside the functions, they are stochastic and
# sampled separately, not part of the deterministic transition/measurement model

# ===============================
# Nonlinear system
# ===============================

def f_nl(x, u):     #non-linear transition function
    return a * np.sin(x + phi_f) + b_p * u

def h_nl(x):        #non-linear measurement function
    return np.sin(c * x + phi_h)


# ===============================
# linear system
# ===============================
# linear approximation: removes phase offsets (phi_f = phi_h = 0)
# and replaces sin(x) with x, removes non-linearity no sin
# IDK maybe keep in here (valid for small x)

#linear transition function (a * x + b_p * u)   #found in the lkf_step
#linear measuremt function (c * x)              #found in the lkf_step

def f_lin(x, u):    # linear transition function
    return a * x + b_p * u
 
def h_lin(x):       # linear measurement function
    return c * x

# linear system matrices
A_lin = np.array([[a]])
C_lin = np.array([[c]])

# ===============================
# Unscented Kalman Filter
# using eigendecomposition for sigma points
# ===============================

def ukf_step(x_hat, P, z, u, f, h, Q, R,uniform=False):
    n = len(x_hat)      # state dimension
    lam = 1.0           # scaling: how far sigma points spread from mean
    n_sig = 2 * n + 1   # total sigma points

    # weights for non-center sigma points (equal weight)
    Wm = np.full(n_sig, 1 / (2*(n + lam)))     # contribution to mean
    Wc = np.full(n_sig, 1 / (2*(n + lam)))     # contribution to covariance
    # center point gets higher weight (closest to current estimate)
    Wm[0] = lam / (n + lam)
    Wc[0] = lam / (n + lam) + (1 - 1.0**2 + 2.0)  # + (1 - alpha^2 + beta)
    # all weights sum to 1

    # --- sigma points via eigendecomposition of P (passed in from previous step) ---
    # eigvals: variance magnitude in each direction
    # eigvecs: directions of variance (natural axes of uncertainty ellipse)
    eigvals, eigvecs = np.linalg.eigh((n + lam) * P)
    S = eigvecs @ np.diag(np.sqrt(np.maximum(eigvals, 0)))  # matrix sqrt of (n+lam)*P

    # center point is current mean estimate
    # symmetric pairs pushed +- along each eigenvector direction, scaled by sqrt(eigenvalue)
    sigma_pts = np.zeros((n, n_sig))
    sigma_pts[:, 0] = x_hat
    for i in range(n):
        sigma_pts[:, i+1]   = x_hat + S[:, i]
        sigma_pts[:, i+1+n] = x_hat - S[:, i]

    # --- prediction ---

    # push each sigma point through nonlinear f    captures how uncertainty transforms
    sigma_pred = np.array([f(sigma_pts[:, i], u) for i in range(n_sig)]).T

    # weighted mean: expected state after propagating through f
    x_pred = sum(Wm[i] * sigma_pred[:, i] for i in range(n_sig))

    # weighted covariance: spread of predicted sigma points around x_pred
    # P from previous step has now been transformed through f into P_pred
    P_pred = Q.copy()   # process noise: irreducible uncertainty added each step
    for i in range(n_sig):
        d = sigma_pred[:, i] - x_pred          # deviation from predicted mean
        P_pred += Wc[i] * np.outer(d, d)       # weighted outer product accumulates spread

    # --- measurement update ---

    # push predicted sigma points through nonlinear h
    z_sig = np.array([h(sigma_pred[:, i]) for i in range(n_sig)])

    # weighted mean of predicted measurements (same logic as x_pred above)
    z_pred = sum(Wm[i] * z_sig[i] for i in range(n_sig))

    # total innovation covariance: predicted measurement spread + sensor noise R
    S_zz = R.copy()
    # cross covariance: correlation between state uncertainty and measurement uncertainty
    # tells kalman gain how to map measurement correction back to state correction
    P_xz = np.zeros((n, len(np.atleast_1d(z_pred))))   # atleast_1d: handles scalar z
    for i in range(n_sig):
        dz = np.atleast_1d(z_sig[i] - z_pred)          # each sigma measurement vs predicted
        dx = sigma_pred[:, i] - x_pred                  # each sigma state vs predicted mean
        S_zz += Wc[i] * np.outer(dz, dz)               # weighted measurement spread
        P_xz += Wc[i] * np.outer(dx, dz)               # weighted state-measurement correlation

    K          = P_xz @ np.linalg.inv(S_zz)                 # Kalman gain: P_xz/S_zz
    innovation = np.atleast_1d(z) - np.atleast_1d(z_pred)   # actual vs predicted measurement
    x_upd      = x_pred + (K @ innovation).flatten()         # corrected state estimate
    P_upd      = P_pred - K @ S_zz @ K.T                    # corrected covariance (uncertainty decreases)

    return x_upd, P_upd, innovation                          # also return innovation for validation

# ===============================
# Extended Kalman Filter
# same structure as UKF/LKF but uses Jacobians instead of sigma points
# ===============================

def ekf_step(x_hat, P, z, u, f, h, Q, R,uniform=False):
    # --- Jacobians evaluated at current estimate ---
    # The Jacobian/slope you need for the linearisation in EKF.
    # It is The partial derivative of f with respect to x, 
    # Basically how does the output of f change as x changes.

 
    # Jacobian matrices evaluated at current estimate (scalar case -> 1x1 matrix)
    F = jacobian(f, 0)(x_hat, u)          # local slope at x_hat, how f stretches uncertainty

    # ---- prediction ----
    # propagate state through nonlinear f (same as UKF, not linearised here)
    x_pred = np.atleast_1d(f(x_hat, u))

    # propagate covariance through linearised F (linear approximation of how spread transforms)
    P_pred = F @ P @ F.T + Q             # same idea a KF but F replaces A: the spread of possible states around x_pred


    # ---- measurement update ----
    H = jacobian(h, 0)(x_pred)    # local slope at x_hat, how h maps uncertainty to measurement

    # predicted measurement through nonlinear h (same as UKF)
    z_pred = np.atleast_1d(h(x_pred))

    S      = H @ P_pred @ H.T + R               # total uncertainty: model spread mapped to meas + sensor noise
    K      = P_pred @ H.T @ np.linalg.inv(S)    # Kalman gain: same formula as LKF, H replaces C

    innovation = np.atleast_1d(z) - z_pred          # actual vs predicted measurement

    # update for state and covariance
    x_upd = x_pred + (K @ innovation).flatten()     
    P_upd = (np.eye(len(x_hat)) - K @ H) @ P_pred   # joseph form

    return x_upd, P_upd, innovation              # also return innovation for validation

# ===============================
# Linear Kalman Filter
# ===============================

def lkf_step(x_hat, P, z, u, A, C, Q, R):
    # ---- prediction ----
    x_pred = A @ x_hat + b_p * np.atleast_1d(u)
    # deterministic state prediction: transition from previous estimate + known input contribution
    P_pred = A @ P @ A.T + Q
    # propagate uncertainty forward through A, then add Q: the spread of possible states around x_pred

    S = C @ P_pred @ C.T + R                # total uncertianty
    K = P_pred @ C.T @ np.linalg.inv(S)     # kalman gain (model uncertianty/total uncerntianty)

    innovation = np.atleast_1d(z) - C @ x_pred

    # update for state and covariance
    x_upd = x_pred + (K @ innovation).flatten()
    P_upd = (np.eye(len(x_hat)) - K @ C) @ P_pred      #joseph form

    return x_upd, P_upd, innovation          # also return innovation for validation

def run_simulation(sim_f, sim_h, label, uniform_noise=False):
    x_true  = np.zeros(N)
    z_meas  = np.zeros(N)
    x_ukf   = np.zeros(N)
    x_ekf   = np.zeros(N)
    x_lkf   = np.zeros(N)

    # innovations stored for validation tests
    inn_ukf = np.zeros(N)
    inn_ekf = np.zeros(N)
    inn_lkf = np.zeros(N)
 
    x_true[0] = x0[0]
    
    # coveriance and state intializiation
    # From earlier set uncertianty covariance to: P0 = np.eye(1) * 1.0 
    P_ukf = P0.copy();  xh_ukf = x0.copy()
    P_ekf = P0.copy();  xh_ekf = x0.copy()
    P_lkf = P0.copy();  xh_lkf = x0.copy()
 
    for k in range(N-1):
 
        # ---- True system (sim_f / sim_h swapped between LSim and NLSim) ----
        u_k = np.sin(2 * np.pi * f_u * time[k])                 # sinusoidal input at frequency f_u
        if uniform_noise:
            w_k = np.sqrt(12 * Q[0,0]) * (np.random.rand() - 0.5)  # U(0,1) -> centered, same variance
        else:
            w_k = np.sqrt(Q[0,0]) * np.random.randn()               # process noise drawn from N(0, Q)
 
        x_true[k+1] = sim_f(np.array([x_true[k]]), u_k)[0] + w_k   # true state evolution
 
        # ---- Sensor output: true state through sim_h + measurement noise ----
        if uniform_noise:
            v_k = np.sqrt(12 * R[0,0]) * (np.random.rand() - 0.5)   # U(0,1) -> centered, same variance
        else:
            v_k = np.sqrt(R[0,0]) * np.random.randn()             # sensor noise drawn from N(0, R)
        z_meas[k] = sim_h(np.array([x_true[k]]))[0] + v_k           # what the sensor delivers
 
        # ---- all three filters receive the same noisy measurement z and input u ----
 
        '''
        # ---- UKF: inputs are ---- :
        # previous estimate
        # covariance
        # measurement
        # input
        # f - non linear transtion function
        # h - non linear measurement function
        # Q - process noise
        # R - measurement noise
        xh_ukf, P_ukf, inn = ukf_step(xh_ukf, P_ukf, z_meas[k], u_k, sim_f, sim_h, Q, R)
        x_ukf[k+1]  = xh_ukf[0]
        inn_ukf[k]  = inn.flatten()[0]
        '''
        
 
        # ---- EKF: same nonlinear f and h as UKF ----
        # but propagates uncertainty through Jacobians (linearisation) instead of sigma points
        xh_ekf, P_ekf, inn = ekf_step(xh_ekf, P_ekf, z_meas[k], u_k, sim_f, sim_h, Q, R)
        x_ekf[k+1]  = xh_ekf[0]
        inn_ekf[k]  = inn.flatten()[0]
 
        # ---- Linear KF: inputs are ---- :
        # same except for different formulas
        xh_lkf, P_lkf, inn = lkf_step(xh_lkf, P_lkf, z_meas[k], u_k, A_lin, C_lin, Q, R)
        x_lkf[k+1]  = xh_lkf[0]
        inn_lkf[k]  = inn.flatten()[0]
 
    return dict(x_true=x_true, z_meas=z_meas,
                x_ukf=x_ukf,   x_ekf=x_ekf,   x_lkf=x_lkf,
                inn_ukf=inn_ukf, inn_ekf=inn_ekf, inn_lkf=inn_lkf,
                label=label)
 

# ===============================
# Validation tests
# ===============================
# basic principle from the slides:
# if the model and parameters are correct, innovations should be white noise
# because the filter has captured all predictable structure -- what remains
# should be pure unpredictable noise, orthogonal to everything past
# if innovations are correlated the signal and noise are not yet orthogonal,
# meaning the model is still missing structure it should have captured


# ==============================================================================
# Whiteness test  ->  returns one row dict
# ==============================================================================

# ----- test 1: run test (sign changes) -- simplest whiteness check ----
# counts how many times the innovation flips sign
# if white noise, sign changes follow B(N-1, 0.5) ~ N((N-1)/2, (N-1)/4)
# too few sign changes means the signal is drifting (correlated), too many means it oscillates (negative correlation).

# ----- test 2: ACF plot + confidence limits -- visual check per lag -----
# plots the autocorrelation at each lag with 95% confidence bands +/- 1.96/sqrt(N)
# under white noise each lag should be ~ N(0, 1/N) so most bars should stay inside the bands
# if bars stick out it means the innovation at time k is still correlated with lag i steps back
# meaning the filter missed something periodic or slowly varying at that timescale

# ----- test 3: portmanteau test -- joint statistical whiteness test across all lags ----
# where the ACF plot gives a visual feel, portmanteau makes it a single statistical statement
# takes the sum of squared ACF values across m lags: N * sum(rho^2(i)) ~ chi2(m)
# if the statistic exceeds the chi2 critical value the null hypothesis of whiteness is rejected
 
def whiteness_test(innovations, label, alpha=0.05):
    e = innovations
    N_e = len(e)
 
    # ---- Run test (sign changes) ----
    signs    = np.sign(e)
    signs_nz = signs[signs != 0]                    # skip exact zeros
    n_used   = len(signs_nz)
    sc       = int(np.sum(np.diff(signs_nz) != 0))  # count sign changes
    mu_sc    = (n_used - 1) / 2
    mu_sc  = (n_used - 1) / 2                   # calculate mean
    var_sc = (n_used - 1) / 4                   # calculate variance
    std_sc = np.sqrt((n_used - 1) / 4)          # standard deviation
    z_a      = norm.ppf(1 - alpha / 2)          # 1.96 for alpha=0.05 (95% confidence)

    # set low,high boundaries expect to be within
    # if fall within variance accept
    low      = mu_sc - z_a * std_sc
    high     = mu_sc + z_a * std_sc
    run_pass = low < sc < high          # when does it pass
 
    # ---- Portmanteau (Box-Pierce) ----
    m   = min(20, N_e // 10)                                            # use either 20 or N/10
    acf = np.array([                                                    # calculate the acf of each
        np.corrcoef(e[:N_e - lag], e[lag:])[0, 1] if lag > 0 else 1.0
        for lag in range(m + 1)
    ])
    conf   = z_a / np.sqrt(N_e)                                         # 95% confidence bounds for individual ACF values, rho_hat(k) ~ N(0, 1/N)
    
    # Large Q => many autocorrelations are collectively too large
    Q_port = N_e * np.sum(acf[1:m + 1] ** 2)
    # Critical value from chi-square distribution
    # Reject H0 if Q exceeds this threshold
    Q_crit = chi2.ppf(1 - alpha, df=m)
    port_pass = Q_port <= Q_crit

    # save result for table and plotting
    return dict(label=label,
                sc=sc, low=int(low), high=int(high), run_pass=run_pass,
                Q_port=Q_port, Q_crit=Q_crit, port_pass=port_pass,
                acf=acf, conf=conf, m=m)
 
# ==============================================================================
# Normality test  ->  appends W, p, pass to existing row dict
# ==============================================================================

# ------ test 4: normal probability plot -- check if innovations are Gaussian -------
# if the filter model is correct and noise is Gaussian, innovations should also be Gaussian
# LKF cannot handle nonlinearity so innovations get skewed away from Gaussian
# EKF partially corrects this via linearisation (Jacobian), 


def normality_test(innovations, row, alpha=0.05):       # simple test if normal using shapiro, same as from matlab with normplot
    e = innovations
    stat_sw, p_sw = shapiro(e[:min(5000, len(e))])
    row['W']         = stat_sw                          # how much look like guassian
    row['p_sw']      = p_sw                             # probality of observing this given it is perfect guassian dataset with this number of samples
    row['norm_pass'] = p_sw > alpha                     # accept if 95% confident interval
    return row
 
# ==============================================================================
# Table printer Simply to give overview of results
# ==============================================================================
 
def print_table(rows):
    # Header
    sep = "-" * 100
    print(sep)
    print(f"{'Combination':<22} | {'Sign chg':>8} {'Range':>14} {'Run':>6} | "
          f"{'Q':>8} {'Qcrit':>7} {'Port':>6} | "
          f"{'W':>7} {'p':>8} {'Norm':>6}")
    print(sep)
    for r in rows:
        run_str  = "PASS" if r['run_pass']  else "FAIL"
        port_str = "PASS" if r['port_pass'] else "FAIL"
        # Normality columns only if computed
        if 'W' in r:
            norm_str = "PASS" if r['norm_pass'] else "FAIL"
            norm_col = f"{r['W']:>7.4f} {r['p_sw']:>8.4f} {norm_str:>6}"
        else:
            norm_col = f"{'---':>7} {'---':>8} {'---':>6}"
        print(f"{r['label']:<22} | {r['sc']:>8} [{r['low']:>4},{r['high']:>4}] {run_str:>6} | "
              f"{r['Q_port']:>8.2f} {r['Q_crit']:>7.2f} {port_str:>6} | "
              f"{norm_col}")
    print(sep)


# ==============================================================================
# 8. Run Gaussian simulations
# ==============================================================================
print("\nRunning LSim  (linear true system, Gaussian noise) ...")
res_lsim  = run_simulation(f_lin, h_lin, "LSim")
 
print("Running NLSim (nonlinear true system, Gaussian noise) ...")
res_nlsim = run_simulation(f_nl,  h_nl,  "NLSim")
 
# Tables over results

# ---- Whiteness and guassian tests table ----
print("\n\n=== Tasks 1 & 2: Whiteness + Normality (Gaussian noise) ===")
rows_gauss = []
for res in [res_lsim, res_nlsim]:
    for flabel, inn_key in [("LKF", "inn_lkf"), ("EKF", "inn_ekf")]:
        combo = f"{res['label']} + {flabel}"
        row   = whiteness_test(res[inn_key], combo)
        normality_test(res[inn_key], row)
        rows_gauss.append(row)
print_table(rows_gauss) 
 
# ---- Plots: ACF + normal prob (Gaussian) ----
for res in [res_lsim, res_nlsim]:
    sim_label = res['label']
 
    # ACF + normal prob plots
    fig2, axes2 = plt.subplots(2, 2, figsize=(13, 12))
    fig2.suptitle(f"Validation — Innovations [{sim_label}] — Gaussian noise", fontsize=13)
 
    sim_idx = 0 if res['label'] == 'LSim' else 1
    for row_idx, (flabel, inn_key) in enumerate([("LKF","inn_lkf"),("EKF","inn_ekf")]):
        e    = res[inn_key]
        r    = rows_gauss[sim_idx * 2 + row_idx]
        N_e  = len(e)
 
        # ACF
        ax_acf = axes2[row_idx, 0]
        lags   = np.arange(1, r['m'] + 1)
        ax_acf.bar(lags, r['acf'][1:r['m']+1], color='steelblue', alpha=0.7)
        ax_acf.axhline( r['conf'], color='red', linestyle='--', lw=1.2,
                        label=f"95% CI (±{r['conf']:.3f})")
        ax_acf.axhline(-r['conf'], color='red', linestyle='--', lw=1.2)
        ax_acf.axhline(0, color='black', lw=0.8)
        ax_acf.plot([], [], ' ', label=f"Q={r['Q_port']:.2f}, Qcrit={r['Q_crit']:.2f}")
        ax_acf.set_title(f'ACF — {flabel}  [{"PASS" if r["port_pass"] else "FAIL"}]')
        ax_acf.set_xlabel('Lag'); ax_acf.set_ylabel('Correlation')
        ax_acf.set_ylim(-0.15, 0.15); ax_acf.legend(fontsize=8); ax_acf.grid(alpha=0.4)
 
        # Normal prob plot
        ax_norm = axes2[row_idx, 1]
        e_s  = np.sort(e)
        p_   = (np.arange(1, N_e + 1) - 0.375) / (N_e + 0.25)
        z_th = norm.ppf(p_)
        ax_norm.scatter(z_th, e_s, s=2, color='steelblue', alpha=0.5, label='Residuals')
        q25, q75 = np.percentile(e_s, [25, 75])
        z25, z75 = norm.ppf([0.25, 0.75])
        slope    = (q75 - q25) / (z75 - z25)
        intercept = q25 - slope * z25
        x_line   = np.array([z_th[0], z_th[-1]])
        ax_norm.plot(x_line, slope * x_line + intercept, 'r-', lw=1.5, label='Normal ref')
        ax_norm.plot([], [], ' ', label=f"W={r['W']:.4f},  p={r['p_sw']:.4f}")
        ax_norm.set_title(f'Normal prob — {flabel}  [{"PASS" if r["norm_pass"] else "FAIL"}]')
        ax_norm.set_xlabel('Theoretical quantiles'); ax_norm.set_ylabel('Sample quantiles')
        ax_norm.legend(fontsize=8); ax_norm.grid(alpha=0.4)
 
    plt.tight_layout()
 
plt.show()   # blocks until all Gaussian plots are closed
 
# ==============================================================================
# 9. Run Uniform simulations
# ==============================================================================
print("\n\nRunning LSim  (linear true system, Uniform noise) ...")
res_lsim_uni  = run_simulation(f_lin, h_lin, "LSim Uniform",  uniform_noise=True)
 
print("Running NLSim (nonlinear true system, Uniform noise) ...")
res_nlsim_uni = run_simulation(f_nl,  h_nl,  "NLSim Uniform", uniform_noise=True)
 
print("\n\n=== Task 3: Whiteness + Normality (Uniform noise) ===")
rows_uni = []
for res in [res_lsim_uni, res_nlsim_uni]:
    for flabel, inn_key in [("LKF", "inn_lkf"), ("EKF", "inn_ekf")]:
        combo = f"{res['label']} + {flabel}"
        row   = whiteness_test(res[inn_key], combo)
        normality_test(res[inn_key], row)
        rows_uni.append(row)
 
print_table(rows_uni)

# ---- Plots: ACF + normal prob (Uniform) ----
for res in [res_lsim_uni, res_nlsim_uni]:
    sim_label = res['label']
    fig, axes = plt.subplots(2, 2, figsize=(13, 12))
    fig.suptitle(f"Validation — Innovations [{sim_label}] — Uniform noise", fontsize=13)
 
    for row_idx, (flabel, inn_key) in enumerate([("LKF","inn_lkf"),("EKF","inn_ekf")]):
        e   = res[inn_key]
        uni_idx = 0 if res['label'] == 'LSim Uniform' else 1
        r   = rows_uni[uni_idx * 2 + row_idx]
        N_e = len(e)
 
        ax_acf = axes[row_idx, 0]
        lags   = np.arange(1, r['m'] + 1)
        ax_acf.bar(lags, r['acf'][1:r['m']+1], color='darkorange', alpha=0.7)
        ax_acf.axhline( r['conf'], color='red', linestyle='--', lw=1.2,
                        label=f"95% CI (±{r['conf']:.3f})")
        ax_acf.axhline(-r['conf'], color='red', linestyle='--', lw=1.2)
        ax_acf.axhline(0, color='black', lw=0.8)
        ax_acf.plot([], [], ' ', label=f"Q={r['Q_port']:.2f}, Qcrit={r['Q_crit']:.2f}")
        ax_acf.set_title(f'ACF — {flabel}  [{"PASS" if r["port_pass"] else "FAIL"}]')
        ax_acf.set_xlabel('Lag'); ax_acf.set_ylabel('Correlation')
        ax_acf.set_ylim(-0.15, 0.15); ax_acf.legend(fontsize=8); ax_acf.grid(alpha=0.4)
 
        ax_norm = axes[row_idx, 1]
        e_s  = np.sort(e)
        p_   = (np.arange(1, N_e + 1) - 0.375) / (N_e + 0.25)
        z_th = norm.ppf(p_)
        ax_norm.scatter(z_th, e_s, s=2, color='darkorange', alpha=0.5, label='Residuals')
        q25, q75 = np.percentile(e_s, [25, 75])
        z25, z75 = norm.ppf([0.25, 0.75])
        slope     = (q75 - q25) / (z75 - z25)
        intercept = q25 - slope * z25
        x_line    = np.array([z_th[0], z_th[-1]])
        ax_norm.plot(x_line, slope * x_line + intercept, 'r-', lw=1.5, label='Normal ref')
        ax_norm.plot([], [], ' ', label=f"W={r['W']:.4f},  p={r['p_sw']:.4f}")
        ax_norm.set_title(f'Normal prob — {flabel}  [{"PASS" if r["norm_pass"] else "FAIL"}]')
        ax_norm.set_xlabel('Theoretical quantiles'); ax_norm.set_ylabel('Sample quantiles')
        ax_norm.legend(fontsize=8); ax_norm.grid(alpha=0.4)
 
    plt.tight_layout()
 
plt.show()   # blocks until all Uniform plots are closed


# ===============================
# Parameter estimation (exercise 5 task 5)
# estimate parameter 'a' in the linear system by augmenting the state
# instead of treating 'a' as a known constant, add it as an extra state
# the EKF then estimates both x and a simultaneously from measurements
# A=I, Sigma=0: a is unknown but constant -- no noise added to it each step
# ===============================
 
# augmented functions: state vector is now [x, a] instead of just [x]
def f_aug(x, u):
    x_state = x[0]          # first element is the actual state
    a_est   = x[1]          # second element is the parameter a being estimated
    return np.array([a_est * x_state + b_p * u,   # now 2D, so upper same as before
                     a_est])                        # a is constant which also need estimatino
 
def h_aug(x):               # same as before now just need to specific for index not a this is only in the underlaying state transition
    return np.array([c * x[0]])   
 
# augmented noise and initial conditions
Q_aug  = np.array([[0.01, 0.0],   # process noise on x (same as before)
                   [0.0,  0.0]])   # no noise on a (constant parameter)
 
P0_aug = np.array([[1.0, 0.0],    # initial identity both equal uncertian
                   [0.0, 1.0]])    
 
x0_aug = np.array([0.0, 0.0])     # initial guess: x=0, a=0
 
# run parameter estimation on LSim data (linear system)
print("\nRunning parameter estimation (augmented EKF on LSim) ...")
x_true_pe  = res_lsim['x_true']   # reuse the same true state from LSim run
z_meas_pe  = res_lsim['z_meas']   # reuse the same measurements
 
x_aug_hist = np.zeros((N, 2))     # store [x, a] estimate at each step
x_aug_hist[0] = x0_aug            # set intial guess

# insert the intial estimations
xh_aug = x0_aug.copy()
P_aug  = P0_aug.copy()
 
# run the same sinusiod
for k in range(N - 1):
    u_k = np.sin(2 * np.pi * f_u * time[k])                                                 # sinusoidal input at frequency f_u
    xh_aug, P_aug, _ = ekf_step(xh_aug, P_aug, z_meas_pe[k], u_k, f_aug, h_aug, Q_aug, R)   #Run the EKF again now with different x,P,f,h,Q
    x_aug_hist[k + 1] = xh_aug          # save estimates

print("="*50)
print("Final results of a estimation")
print("="*50)
print(f"  True a          = {a}")
print(f"  Estimated a     = {x_aug_hist[-1, 1]:.4f}")
print(f"  Initial guess a = {x0_aug[1]}")
 
 
# plot parameter estimation result
fig, axes = plt.subplots(2, 1, figsize=(12, 6))
fig.suptitle("Parameter Estimation — Augmented EKF on LSim (estimating a)")
 
axes[0].plot(time, x_true_pe,          label='True state')
axes[0].plot(time, x_aug_hist[:, 0], '--', label='EKF estimate of x')
axes[0].set_title('State estimate'); axes[0].legend(); axes[0].grid()
 
axes[1].plot(time, x_aug_hist[:, 1], label='Estimated a')
axes[1].axhline(a, color='red', linestyle='--', label=f'True a = {a}')
axes[1].set_title('Parameter estimate: a'); axes[1].legend(); axes[1].grid()
 
plt.tight_layout()
plt.savefig("results/parameter_estimation.png", dpi=150, bbox_inches='tight')
 
plt.show()