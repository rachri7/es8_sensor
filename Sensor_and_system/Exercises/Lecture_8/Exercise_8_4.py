import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Parameters
gamma = 1.74        # exponent coeffient in path loss, higher more signal decay with distance
f     = 2.45e9      # frequency
c     = 3e8         # speed of light
A     = -10 * gamma * np.log10(4 * np.pi * f / c)   # TxPower A: RSSI at 1 m

room_w, room_h = 5.0, 10.0
beacons = np.array([[0.,0.],[room_w,0.],[0.,room_h],[room_w,room_h]])   # place beacons
sigma_rssi = 3.0                                                        # noise on each RSSI reading
# Becomes relative higher ratio when longer distance


def distance_to_rssi(d):    # go from distance to RSSI
    return -10 * gamma * np.log10(np.maximum(d, 0.01)) + A

def rssi_to_distance(rssi): # Go from RSSI to Distance
    return 10 ** ((A - rssi) / (10 * gamma))

def cost(pos, beacon_pos, distances):  # Optimation formulation
    """Minimise sum of squared residuals between geometric and estimated distances."""
    # pos            = candidate (x, y) position
    # beacon_pos     = known beacon coordinates (bx_i, by_i)
    # distances      = RSSI-estimated distances 
    # sqrt(...)      = geometric distance from candidate pos to beacon i
    # sum gives how far each of the circles the beacons create are from the x,y points
    return sum((np.sqrt((pos[0]-bx)**2+(pos[1]-by)**2) - d)**2
               for (bx,by), d in zip(beacon_pos, distances))

def trilaterate(beacon_pos, distances):     # minimise the optimisation problem
    x0 = np.array([room_w/2, room_h/2])    # initial guess: room centre
    return minimize(cost, x0, args=(beacon_pos, distances),
                    method='Nelder-Mead',
                    options={'xatol':1e-6,    # stop when all tested candidate points
                                              # are within 1e-6 m of each other
                             'fatol':1e-6,    # stop when J cost at those candidates
                                              # differs by less than 1e-6
                             'maxiter':10000}).x  # hard cap on iterations
# .x means extract just the solution array (the optimal (x, y) values) 


np.random.seed(42)

# --- Three test positions, 40 trials each ---
test_positions = [
    np.array([1.25, 2.50]),
    np.array([2.50, 5.00]),
    np.array([3.75, 7.50]),
]

all_estimates = []
all_errors    = []

for tp in test_positions:
    ests = []
    for _ in range(40):
        true_d = np.array([np.linalg.norm(tp - b) for b in beacons])    # calculate distance from determinstic point to each beacon
        noisy  = distance_to_rssi(true_d) + np.random.normal(0, sigma_rssi, 4)  # add noise and convert dist to RSSI for all 4 each noise independent
        d_meas = rssi_to_distance(noisy)                                # convert back to get dist
        ests.append(trilaterate(beacons, d_meas))                       # all estimation using optimisatin problem new one each time
    ests = np.array(ests)
    all_estimates.append(ests)
    all_errors.append(np.linalg.norm(ests - tp, axis=1))                # the error of estimate compared to true position

# --- Figure ---
fig, axes = plt.subplots(1, 4, figsize=(16, 7),
                         gridspec_kw={'width_ratios': [1,1,1,1.2]})
fig.suptitle(f"BLE Trilateration  –  5×10 m room, 4 corner beacons\n"
             f"(γ={gamma}, σ_RSSI={sigma_rssi} dBm, 40 trials per location)",
             fontsize=12, fontweight='bold')

for col, (tp, ests, errs) in enumerate(zip(test_positions, all_estimates, all_errors)): # for each test positino we plot this
    ax = axes[col]

    # Room outline
    rect = plt.Polygon([[0,0],[room_w,0],[room_w,room_h],[0,room_h]],
                        fill=False, edgecolor='black', linewidth=2)
    ax.add_patch(rect)

    # Beacons
    ax.scatter(beacons[:,0], beacons[:,1], marker='^', s=120,
               color='royalblue', zorder=5)
    for i,(bx,by) in enumerate(beacons):
        ax.annotate(f'B{i+1}', (bx,by), xytext=(5,4),
                    textcoords='offset points', fontsize=8, color='royalblue')

    # Estimated positions (red)
    ax.scatter(ests[:,0], ests[:,1], s=35, color='red',
               alpha=0.7, zorder=4, label='Estimated')

    # True position (green)
    ax.scatter(*tp, s=140, color='limegreen', zorder=6,
               edgecolors='black', linewidth=0.8, label='True position')

    ax.set_xlim(-0.5, room_w+0.5)
    ax.set_ylim(-0.5, room_h+0.5)
    ax.set_aspect('equal')
    ax.set_xlabel('x [m]')
    if col == 0:
        ax.set_ylabel('y [m]')
        ax.legend(fontsize=8, loc='upper right')
    ax.set_title(f"True: ({tp[0]:.2f}, {tp[1]:.2f}) m\n"
                 f"avg={errs.mean():.2f} m  90th={np.percentile(errs,90):.2f} m",
                 fontsize=9)
    ax.grid(True, alpha=0.3)

# --- Error histogram (all 120 trials pooled) ---
ax4 = axes[3]
all_err_flat = np.concatenate(all_errors)

ax4.hist(all_err_flat, bins=20, color='steelblue', edgecolor='white')
ax4.axvline(all_err_flat.mean(), color='red', lw=2,
            label=f'Mean  {all_err_flat.mean():.2f} m')
ax4.axvline(np.percentile(all_err_flat,90), color='green', lw=2, linestyle='--',
            label=f'90th  {np.percentile(all_err_flat,90):.2f} m')
ax4.set_xlabel('Position error [m]')
ax4.set_ylabel('Count')
ax4.set_title('Error distribution\n(all locations, 120 trials)')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
print("Done")