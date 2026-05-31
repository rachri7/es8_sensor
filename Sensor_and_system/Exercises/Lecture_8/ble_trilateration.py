"""
BLE Indoor Positioning via RSSI-based Trilateration
====================================================
Follows Bembenik & Falcman (2020) – JoWUA 11(3):50-69
DOI: 10.22667/JOWUA.2020.09.30.050

Room: 5 x 10 m, beacons at each corner.
γ = 1.74  (close to paper's measured 1.736)
f = 2.45 GHz (mid-band BLE)

Signal model (Eq. 8 in paper):
    RSSI = -10·γ·log10(d) + A
  ⟹  d = 10^((A - RSSI) / (10·γ))          (Eq. 9)

TxPower A is calibrated at 1 m; here we set it from the physics:
    A = 10·γ·log10(4πf/c)   [see Eq. 7 in paper, first term = 0 at d=1]

Position estimated by minimising
    J(x,y) = Σ_i [dist(x,y, beacon_i) - d̂_i]²
using scipy.optimize.minimize (Nelder-Mead / L-BFGS-B).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.optimize import minimize

# Directory where the script lives

# ── Physical / model parameters ────────────────────────────────────────────
GAMMA = 1.74          # path-loss exponent (paper measured 1.736 indoors)
F_HZ  = 2.45e9        # BLE centre frequency [Hz]
C     = 3e8           # speed of light [m/s]

# TxPower A: RSSI measured at exactly 1 m in free space
# From Eq. 7:  FSPL(dB) = 10γ log10(d) + 10γ log10(4πf/c)
# At d=1 m the first term is zero, so  A = -10γ log10(4πf/c)
# (negative because RSSI is a received power, i.e. negative of FSPL)
A_TXPOWER = -10 * GAMMA * np.log10(4 * np.pi * F_HZ / C)  # ≈ -59 dBm

# ── Room & beacon layout ────────────────────────────────────────────────────
ROOM_W, ROOM_H = 5.0, 10.0   # metres

BEACONS = np.array([
    [0.0,    0.0   ],   # bottom-left
    [ROOM_W, 0.0   ],   # bottom-right
    [0.0,    ROOM_H],   # top-left
    [ROOM_W, ROOM_H],   # top-right
])

N_BEACONS = len(BEACONS)

# ── RSSI model helpers ──────────────────────────────────────────────────────

def rssi_from_distance(d: float, sigma_noise: float = 0.0) -> float:
    """Ideal RSSI [dBm] at distance d, plus optional Gaussian noise."""
    if d < 0.01:
        d = 0.01
    rssi = -10 * GAMMA * np.log10(d) + A_TXPOWER
    if sigma_noise > 0:
        rssi += np.random.normal(0, sigma_noise)
    return rssi


def distance_from_rssi(rssi: float) -> float:
    """Invert the log-distance model: d = 10^((A - RSSI)/(10·γ))  [Eq. 9]"""
    return 10 ** ((A_TXPOWER - rssi) / (10 * GAMMA))


# ── Trilateration (optimisation-based) ─────────────────────────────────────

def cost_function(pos, beacons, d_measured):
    """
    Residual sum of squares between measured distances and Euclidean
    distances from candidate position to each beacon.
    """
    x, y = pos
    total = 0.0
    for i, (bx, by) in enumerate(beacons):
        d_geom = np.sqrt((x - bx)**2 + (y - by)**2)
        total += (d_geom - d_measured[i])**2
    return total


def estimate_position(rssi_measurements, beacons,
                      x0=None, method="L-BFGS-B"):
    """
    Given a list of RSSI values (one per beacon), return the (x, y)
    position estimate via optimisation.

    Parameters
    ----------
    rssi_measurements : array-like, shape (N,)
    beacons           : array-like, shape (N, 2)
    x0                : initial guess [x, y]; defaults to room centre
    method            : scipy optimiser name

    Returns
    -------
    pos_est : np.ndarray shape (2,)
    result  : scipy OptimizeResult
    """
    d_meas = np.array([distance_from_rssi(r) for r in rssi_measurements])

    if x0 is None:
        x0 = np.array([ROOM_W / 2, ROOM_H / 2])

    bounds = [(0, ROOM_W), (0, ROOM_H)]

    result = minimize(
        cost_function,
        x0,
        args=(beacons, d_meas),
        method=method,
        bounds=bounds,
        options={"maxiter": 5000, "ftol": 1e-12}
    )
    return result.x, result


# ── Simulation ──────────────────────────────────────────────────────────────

def simulate(true_pos, n_trials=40, sigma_noise=3.0, seed=0):
    """
    Simulate n_trials RSSI measurements at true_pos, each with independent
    Gaussian noise on each beacon's RSSI, and return the estimated positions.

    sigma_noise : std-dev of RSSI noise in dBm
                  (paper observed ≈ 3-5 dBm fluctuation indoors)
    """
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(n_trials):
        rssi = []
        for bx, by in BEACONS:
            d_true = np.sqrt((true_pos[0]-bx)**2 + (true_pos[1]-by)**2)
            noise  = rng.normal(0, sigma_noise)
            rssi.append(rssi_from_distance(d_true) + noise)
        pos_est, _ = estimate_position(rssi, BEACONS)
        estimates.append(pos_est)
    return np.array(estimates)


# ── Metrics ─────────────────────────────────────────────────────────────────

def positioning_error(true_pos, estimates):
    errors = np.linalg.norm(estimates - true_pos, axis=1)
    return {
        "min":   errors.min(),
        "max":   errors.max(),
        "mean":  errors.mean(),
        "p90":   np.percentile(errors, 90),
        "std":   errors.std(),
    }


# ── Plotting ────────────────────────────────────────────────────────────────

def plot_room(true_positions_and_estimates, title="BLE Trilateration – 5×10 m room"):
    """
    Replicate the style of slide 5 / Figure 5 in the paper:
      green dot  = real position
      red dots   = estimated positions from noisy RSSI
    """
    fig, axes = plt.subplots(1, len(true_positions_and_estimates),
                             figsize=(5 * len(true_positions_and_estimates), 6))
    if len(true_positions_and_estimates) == 1:
        axes = [axes]

    for ax, (true_pos, estimates) in zip(axes, true_positions_and_estimates):
        # Room outline
        room = plt.Polygon(
            [[0,0],[ROOM_W,0],[ROOM_W,ROOM_H],[0,ROOM_H]],
            fill=False, edgecolor="black", linewidth=2
        )
        ax.add_patch(room)

        # Beacons
        ax.scatter(BEACONS[:,0], BEACONS[:,1],
                   s=120, marker="^", color="royalblue",
                   zorder=5, label="Beacon")
        for i, (bx, by) in enumerate(BEACONS):
            ax.annotate(f"B{i+1}", (bx, by),
                        textcoords="offset points", xytext=(5, 5),
                        fontsize=8, color="royalblue")

        # RSSI circles (mean estimated distance from each beacon)
        mean_est = estimates.mean(axis=0)
        for bx, by in BEACONS:
            r = np.sqrt((mean_est[0]-bx)**2 + (mean_est[1]-by)**2)
            circle = plt.Circle((bx, by), r,
                                 fill=False, linestyle="--",
                                 color="steelblue", alpha=0.25, linewidth=1)
            ax.add_patch(circle)

        # Estimated positions (red dots – like slide 5)
        ax.scatter(estimates[:,0], estimates[:,1],
                   s=30, color="red", alpha=0.6, zorder=4, label="Estimated")

        # True position (green dot – like slide 5)
        ax.scatter(*true_pos, s=120, color="limegreen", zorder=6,
                   edgecolors="black", linewidth=0.8, label="True position")

        # Error stats
        err = positioning_error(true_pos, estimates)
        ax.set_title(
            f"True: ({true_pos[0]:.1f}, {true_pos[1]:.1f}) m\n"
            f"avg err={err['mean']:.2f} m  90th={err['p90']:.2f} m",
            fontsize=9
        )
        ax.set_xlim(-0.5, ROOM_W + 0.5)
        ax.set_ylim(-0.5, ROOM_H + 0.5)
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_error_vs_noise():
    """Show how positioning error grows with RSSI noise level."""
    sigmas = np.linspace(0.5, 8, 16)
    true_pos = np.array([2.5, 5.0])
    means, p90s = [], []
    for sig in sigmas:
        ests = simulate(true_pos, n_trials=200, sigma_noise=sig, seed=42)
        e = positioning_error(true_pos, ests)
        means.append(e["mean"])
        p90s.append(e["p90"])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sigmas, means, "o-", label="Mean error")
    ax.plot(sigmas, p90s, "s--", label="90th percentile")
    ax.axhline(1.75, color="gray", linestyle=":", linewidth=1,
               label="Paper avg (1.75 m)")
    ax.set_xlabel("RSSI noise σ [dBm]")
    ax.set_ylabel("Positioning error [m]")
    ax.set_title("Positioning error vs. RSSI noise\n(γ=1.74, 5×10 m room, 4 corner beacons, 200 trials)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("BLE Trilateration – Python implementation")
    print(f"  γ = {GAMMA},  f = {F_HZ/1e9} GHz")
    print(f"  TxPower A = {A_TXPOWER:.2f} dBm  (at 1 m)")
    print(f"  Room: {ROOM_W} × {ROOM_H} m,  {N_BEACONS} corner beacons")
    print("=" * 60)

    # ── Three test positions (mirroring the paper's multi-location evaluation)
    test_positions = [
        np.array([1.25, 2.50]),   # near bottom-left quadrant
        np.array([2.50, 5.00]),   # room centre
        np.array([3.75, 7.50]),   # near top-right quadrant
    ]

    all_results = []
    for tp in test_positions:
        ests = simulate(tp, n_trials=40, sigma_noise=3.0)
        err  = positioning_error(tp, ests)
        all_results.append((tp, ests))
        print(f"\nTrue position ({tp[0]:.2f}, {tp[1]:.2f}) m:")
        print(f"  min error  = {err['min']:.3f} m")
        print(f"  max error  = {err['max']:.3f} m")
        print(f"  avg error  = {err['mean']:.3f} m")
        print(f"  90th pct   = {err['p90']:.3f} m")

    # ── Figure 1: Room floorplan (slide-5 style)
    fig1 = plot_room(all_results,
                     title="BLE Trilateration – 5×10 m room, 4 corner beacons\n"
                           "(γ=1.74, σ_RSSI=3 dBm, 40 trials per location)")
    fig1.savefig("ble_trilateration_room.png",
                 dpi=150, bbox_inches="tight")
    print("\nSaved: ble_trilateration_room.png")

    # ── Figure 2: Error vs noise
    fig2 = plot_error_vs_noise()
    fig2.savefig("ble_error_vs_noise.png",
                 dpi=150, bbox_inches="tight")
    print("Saved: ble_error_vs_noise.png")

    # ── Also save the Python script itself
    import shutil
    shutil.copy(__file__, "ble_trilateration.py")
    print("Saved: ble_trilateration.py")

    plt.show()