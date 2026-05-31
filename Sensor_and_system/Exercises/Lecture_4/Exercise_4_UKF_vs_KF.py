import numpy as np
import matplotlib.pyplot as plt

# ===============================
# Parameters from exercise 4
# ===============================
a     = 0.95              # AR coefficient, < 1 for stability
k_    = 1                 # desired steady-state gain       ,settles at exactly x* = k*u.
b_p   = k_ * (1 - a)     # input gain: derived from k and it balances how much the new input is weighted against the decay from a

# sin parameters ...
c     = 1                # measurement scaling, analogous to H in linear KF (scales non-linearity term inside the sin)

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

# linear system matrices
A_lin = np.array([[a]])
C_lin = np.array([[c]])

# ===============================
# Unscented Kalman Filter
# using eigendecomposition for sigma points
# ===============================

def ukf_step(x_hat, P, z, u, f, h, Q, R):
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

    K         = P_xz @ np.linalg.inv(S_zz)                 # Kalman gain: P_xz/S_zz
    innovation = np.atleast_1d(z) - np.atleast_1d(z_pred)  # actual vs predicted measurement
    x_upd     = x_pred + (K @ innovation).flatten()         # corrected state estimate
    P_upd     = P_pred - K @ S_zz @ K.T                    # corrected covariance (uncertainty decreases)

    return x_upd, P_upd                                     # become x_hat and P for next step

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

    return x_upd, P_upd

# ===============================
# Storage
# ===============================

x_true   = np.zeros(N)
z_meas   = np.zeros(N)

x_ukf    = np.zeros(N)
x_lkf    = np.zeros(N)

x_true[0] = x0[0]

P_ukf = P0.copy()
P_lkf = P0.copy()

xh_ukf = x0.copy()
xh_lkf = x0.copy()

# ===============================
# Simulation + filters
# ===============================

# run for each iteration
for k in range(N-1):

    # ---- True nonlinear system ----
    u_k = np.sin(2 * np.pi * f_u * time[k])        # sinusoidal input at frequency f_u
    w_k = np.sqrt(Q[0,0]) * np.random.randn()       # process noise drawn from N(0, Q)

    x_true[k+1] = f_nl(np.array([x_true[k]]), u_k)[0] + w_k   # true state evolution

    # ---- Sensor output: true state through nonlinear h + measurement noise ----
    v_k       = np.sqrt(R[0,0]) * np.random.randn()            # sensor noise drawn from N(0, R)
    z_meas[k] = h_nl(np.array([x_true[k]]))[0] + v_k           # what the sensor delivers

    # ---- both filters receive same noisy measurement z and input u ----

    # ---- UKF: inputs are ---- :
    # previous estimate
    # covariance
    # measurement
    # input
    # f - non linear transtion function
    # h - non linear measurement function
    # Q - process noise
    # R - measurement noise
    xh_ukf, P_ukf = ukf_step(xh_ukf, P_ukf, z_meas[k], u_k, f_nl, h_nl, Q, R)
    x_ukf[k+1]    = xh_ukf[0]

    # ---- Linear KF: inputs are ---- :
    # same expect for different formulas
    #
    xh_lkf, P_lkf = lkf_step(xh_lkf, P_lkf, z_meas[k], u_k, A_lin, C_lin, Q, R)
    x_lkf[k+1]    = xh_lkf[0]

# ===============================
# Plot Results
# ===============================

plt.figure(figsize=(12, 8))
plt.suptitle("Lecture 4 --- UKF vs Linear KF on Nonlinear System")

plt.subplot(3, 1, 1)
plt.plot(time, x_true,  label='True state')
plt.plot(time, x_ukf,  '--', label='UKF estimate')
plt.plot(time, x_lkf,  ':',  label='Linear KF estimate')
plt.title('State estimate')
plt.legend(); plt.grid()

plt.subplot(3, 1, 2)
plt.plot(time, z_meas, label='Measurement $z_k$', alpha=0.5)
plt.title('Measurements')
plt.legend(); plt.grid()

plt.subplot(3, 1, 3)
plt.plot(time, x_true - x_ukf,  label='UKF error')
plt.plot(time, x_true - x_lkf, '--', label='Linear KF error')
plt.title('Estimation error')
plt.legend(); plt.grid()

plt.tight_layout()
plt.show()