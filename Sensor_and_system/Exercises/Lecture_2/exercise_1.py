"""
Lecture 2 - Exercise 1
Sampling the piecewise PDF, live plot of histogram vs true PDF,
running sample mean and variance converging to theoretical values.

True parameters:  mu = 2,  sigma^2 = 67/50 = 1.34
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Piecewise PDF ──────────────────────────────────────────────────────────────
def pdf(x):
    """
    f(x) = (3/5)x                    for 0 < x <= 1
           (3/5)(x-2)^2              for 1 < x <= 3   [= (sqrt(3)x - sqrt(12))^2 / 5]
           12/5 - (3/5)x             for 3 < x <= 4
           0                         otherwise
    """
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    m1 = (x > 0)  & (x <= 1)
    m2 = (x > 1)  & (x <= 3)
    m3 = (x > 3)  & (x <= 4)
    y[m1] = (3/5) * x[m1]
    y[m2] = (3/5) * (x[m2] - 2)**2   # simplified form of (sqrt(3)*x - sqrt(12))^2 / 5
    y[m3] = 12/5 - (3/5) * x[m3]
    return y

# ── Inverse CDF sampling ───────────────────────────────────────────────────────
# CDF pieces (derived analytically):
#   F(x) = (3/10)x^2                             for 0 < x <= 1  [F(1) = 3/10]
#   F(x) = 3/10 + [(x-2)^3 + 1] / 5             for 1 < x <= 3  [F(3) = 7/10]
#   F(x) = [24x - 3x^2 - 38] / 10               for 3 < x <= 4  [F(4) = 1]
#
# Inverting each piece gives the formulas below.

def sample_icdf(u):
    """Map uniform samples u in [0,1] to samples from f(x) via inverse CDF."""
    u = np.asarray(u, dtype=float)
    x = np.empty_like(u)

    # Piece 1: u <= 3/10  =>  x = sqrt(10u / 3)
    m1 = u <= 3/10
    x[m1] = np.sqrt(10 * u[m1] / 3)

    # Piece 2: 3/10 < u <= 7/10  =>  x = 2 + cbrt(5u - 5/2)
    # (cbrt handles negative argument correctly for the left half of the piece)
    m2 = (u > 3/10) & (u <= 7/10)
    x[m2] = 2 + np.cbrt(5 * u[m2] - 5/2)

    # Piece 3: u > 7/10  =>  x = 4 - sqrt(120(1-u)) / 6
    m3 = u > 7/10
    x[m3] = 4 - np.sqrt(120 * (1 - u[m3])) / 6

    return x

# ── True parameters ────────────────────────────────────────────────────────────
TRUE_MEAN = 2.0
TRUE_VAR  = 67 / 50   # 1.34

BATCH_SIZE = 20       # new samples drawn per frame
X_PLOT = np.linspace(0, 4.2, 600)

# ── Figure layout ──────────────────────────────────────────────────────────────
plt.ion()
fig = plt.figure(figsize=(13, 7))
fig.suptitle("Exercise 1 — Live sampling of piecewise PDF", fontsize=13, fontweight='bold')

gs = gridspec.GridSpec(2, 2, width_ratios=[1.6, 1], hspace=0.45, wspace=0.35)
ax_hist = fig.add_subplot(gs[:, 0])   # histogram occupies full left column
ax_mean = fig.add_subplot(gs[0, 1])   # top right: running mean
ax_var  = fig.add_subplot(gs[1, 1])   # bottom right: running variance

# ── Histogram axes (static elements drawn once) ────────────────────────────────
ax_hist.set_xlabel("$x$", fontsize=11)
ax_hist.set_ylabel("Density", fontsize=11)
ax_hist.set_title("Histogram vs true PDF $f(x)$")
ax_hist.set_xlim(-0.1, 4.3)
ax_hist.set_ylim(0, 0.72)

# ── Running statistics axes ────────────────────────────────────────────────────
ax_mean.axhline(TRUE_MEAN, color='royalblue', linestyle='--', lw=1.8,
                label=f'True $\\mu = {TRUE_MEAN}$')
ax_mean.set_ylabel("Mean", fontsize=10)
ax_mean.set_title("Running sample mean")
ax_mean.set_ylim(0.5, 3.5)
ax_mean.legend(fontsize=8)

ax_var.axhline(TRUE_VAR, color='royalblue', linestyle='--', lw=1.8,
               label=f'True $\\sigma^2 = {TRUE_VAR}$')
ax_var.set_xlabel("Number of samples", fontsize=10)
ax_var.set_ylabel("Variance", fontsize=10)
ax_var.set_title("Running sample variance")
ax_var.set_ylim(0, 3.0)
ax_var.legend(fontsize=8)

# ── Sampling state ─────────────────────────────────────────────────────────────
samples   = []
run_means = []
run_vars  = []
ns        = []

# ── Live sampling loop ─────────────────────────────────────────────────────────
print("Close the plot window to stop.")

while plt.fignum_exists(fig.number):

    # --- draw a batch of new samples via inverse CDF -------------------------
    u_batch = np.random.uniform(0, 1, BATCH_SIZE)
    new_samples = sample_icdf(u_batch)
    samples.extend(new_samples.tolist())

    arr = np.array(samples)
    n   = len(arr)

    # --- update running statistics -------------------------------------------
    mu_hat  = arr.mean()
    var_hat = arr.var(ddof=1)   # unbiased estimator (divide by n-1)
    run_means.append(mu_hat)
    run_vars.append(var_hat)
    ns.append(n)

    # --- redraw histogram panel ----------------------------------------------
    ax_hist.cla()
    ax_hist.hist(arr, bins=50, density=True, alpha=0.45,
                 color='steelblue', label=f'$n = {n}$ samples')
    ax_hist.plot(X_PLOT, pdf(X_PLOT), 'k-', lw=2, label='True PDF $f(x)$')
    # true expectation — dashed blue vertical line
    ax_hist.axvline(TRUE_MEAN, color='royalblue', linestyle='--', lw=2.0,
                    label=f'True $\\mu = {TRUE_MEAN}$')
    # sample mean — solid red vertical line
    ax_hist.axvline(mu_hat, color='tomato', linestyle='-', lw=2.0,
                    label=f'Sample mean $\\hat{{\\mu}} = {mu_hat:.3f}$')
    ax_hist.set_xlim(-0.1, 4.3)
    ax_hist.set_ylim(0, 0.72)
    ax_hist.set_xlabel("$x$", fontsize=11)
    ax_hist.set_ylabel("Density", fontsize=11)
    ax_hist.set_title("Histogram vs true PDF $f(x)$")
    ax_hist.legend(fontsize=9, loc='upper left')

    # --- update running mean line --------------------------------------------
    # clear and redraw so the x-axis expands with new samples
    ax_mean.cla()
    ax_mean.axhline(TRUE_MEAN, color='royalblue', linestyle='--', lw=1.8,
                    label=f'True $\\mu = {TRUE_MEAN}$')
    ax_mean.plot(ns, run_means, color='tomato', lw=1.2, label='Sample mean')
    ax_mean.set_ylabel("Mean", fontsize=10)
    ax_mean.set_title("Running sample mean")
    ax_mean.set_ylim(0.5, 3.5)
    ax_mean.legend(fontsize=8)

    # --- update running variance line ----------------------------------------
    ax_var.cla()
    ax_var.axhline(TRUE_VAR, color='royalblue', linestyle='--', lw=1.8,
                   label=f'True $\\sigma^2 = {TRUE_VAR}$')
    ax_var.plot(ns, run_vars, color='seagreen', lw=1.2, label='Sample variance')
    ax_var.set_xlabel("Number of samples", fontsize=10)
    ax_var.set_ylabel("Variance", fontsize=10)
    ax_var.set_title("Running sample variance")
    ax_var.set_ylim(0, 3.0)
    ax_var.legend(fontsize=8)

    # --- print progress every 500 samples ------------------------------------
    if n % 500 < BATCH_SIZE:
        print(f"n={n:5d}  sample mean={mu_hat:.4f}  sample var={var_hat:.4f}")

    plt.pause(0.05)   # short pause — hands control back to the GUI event loop

plt.ioff()
print("Done.")
