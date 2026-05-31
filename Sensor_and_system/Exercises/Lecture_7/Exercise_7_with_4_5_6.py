import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import minimize

# ===============================
# Parameters
# ===============================
N_rays       = 360           # number of LiDAR beams
lidar_range  = 20.0          # max LiDAR range [m]
sigma_d      = 0.05          # range noise std [m]
sigma_alpha  = np.deg2rad(1) # angle noise std [rad]

# ===============================
# Exercise 3: intersection check functions
# ===============================

def check_circle_intersection(angle, lidar_pos, robot_pos, robot_radius):
    """
    Check if a ray from lidar_pos at given angle hits
    cylindrical robot at robot_pos with robot_radius.
    Returns hit point (x,y) or None.
    """
    # ray direction unit vector from angle
    dx = np.cos(angle)
    dy = np.sin(angle)

    # the distant between the two
    fx = lidar_pos[0] - robot_pos[0]
    fy = lidar_pos[1] - robot_pos[1]

    # just splitting splittin into quadratic, so can solve for t with quadratic formula
    a_c = dx**2 + dy**2                     # dependent on sqrt t
    b_c = 2*(fx*dx + fy*dy)                 # dependent on t
    c_c = fx**2 + fy**2 - robot_radius**2   # Not dependent on t

    disc = b_c**2 - 4*a_c*c_c
    if disc < 0:
        return None     # no intersection

    # solving quadratic formula
    t = (-b_c - np.sqrt(disc)) / (2*a_c)   # closest intersection
    if t <= 0 or t > lidar_range:
        return None     # behind lidar or out of range

    x_hit = lidar_pos[0] + t*dx     # where hit on x 
    y_hit = lidar_pos[1] + t*dy     # where hit on y
    return (x_hit, y_hit)

def check_wall_intersection(angle, lidar_pos, wall_start, wall_end):
    """
    Check if a ray from lidar_pos at given angle hits a wall segment
    defined from wall_start to wall_end.
    Returns hit point in Cartesian (x,y) or None.
    """

    # ray direction unit vector from angle
    dx = np.cos(angle)
    dy = np.sin(angle)

    # wall direction vector
    wx = wall_end[0] - wall_start[0]
    wy = wall_end[1] - wall_start[1]


    # 2D cross product of ray direction and wall direction
    # = 0 means they are parallel (never intersect) 
    denom = dx*wy - dy*wx
    if abs(denom) < 1e-9:
        return None     # ray parallel to wall



    # t = how far along the ray (must be > 0 and < lidar_range)
    t = ((wall_start[0] - lidar_pos[0])*wy -
         (wall_start[1] - lidar_pos[1])*wx) / denom
    # s = fractional position along wall vector (0 = wall_start, 1 = wall_end)
    # outside [0,1] means ray never hit wall segment
    s = ((wall_start[0] - lidar_pos[0])*dy -
         (wall_start[1] - lidar_pos[1])*dx) / denom

    if t <= 0 or t > lidar_range:
        return None     # behind lidar or out of range
    if s < 0 or s > 1:
        return None     # outside wall segment

    # return where they hit just start and then unit vector for x and y scaling
    x_hit = lidar_pos[0] + t*dx
    y_hit = lidar_pos[1] + t*dy
    return (x_hit, y_hit)

def add_noise(x_hit, y_hit, lidar_pos):
    """
    Add noise in polar domain (range + angle) then convert back to Cartesian.
    nu_d ~ N(0, sigma_d^2), nu_alpha ~ N(0, sigma_alpha^2)
    """
    # vector from lidar to hit point
    dx = x_hit - lidar_pos[0]
    dy = y_hit - lidar_pos[1]

    # convert hit point to polar coordinates relative to lidar
    d_true     = np.sqrt(dx**2 + dy**2)  # true range: Pythagoras distance to hit
    alpha_true = np.arctan2(dy, dx)      # true bearing: angle of hit from lidar

    # add independent Gaussian noise to range and angle (sensor noise model)
    d_noisy     = d_true     + np.random.normal(0, sigma_d)      # ~5 cm range noise
    alpha_noisy = alpha_true + np.random.normal(0, sigma_alpha)  # ~1 deg angle noise

    # convert noisy polar back to Cartesian:
    x_noisy = lidar_pos[0] + d_noisy * np.cos(alpha_noisy)
    y_noisy = lidar_pos[1] + d_noisy * np.sin(alpha_noisy)
    # remember tangentail axis, angle error scale with difference
    # radial axis just fixed
    return x_noisy, y_noisy

# ===============================
# Exercise 4: LiDAR shooting at cylinder
# ===============================

# Place our Robot and Lidar
lidar_pos = np.array([0.0, 0.0])
robot_pos    = np.array([10.0, 0.0])
robot_radius = 1.5

# Look over all N_rayes here 360 so one per degree
angles = np.linspace(0, 2*np.pi, N_rays, endpoint=False)

# for each beam: check if it hits the robot
robot_pts = []
for angle in angles:
    pt = check_circle_intersection(angle, lidar_pos, robot_pos, robot_radius)
    if pt is not None:  # add noise to sample only when hit robot
        xn, yn = add_noise(pt[0], pt[1], lidar_pos)
        robot_pts.append((xn, yn))

robot_pts = np.array(robot_pts)
fig, ax = plt.subplots(figsize=(7, 7))
ax.add_patch(patches.Circle((robot_pos), robot_radius, color='steelblue',
             fill=False, lw=2, label='Cylindrical robot'))
ax.scatter(robot_pts[:,0], robot_pts[:,1], s=10,
           color='red', label='LiDAR point cloud')
ax.scatter(*lidar_pos, color='blue', s=80,
           marker='^', label='LiDAR origin', zorder=5)
ax.set_aspect('equal'); ax.grid(); ax.legend()
ax.set_title('Exercise 4 --- LiDAR on cylindrical robot (with noise)')
plt.tight_layout()
plt.show()

# ===============================
# Exercise 5: Robot in front of wall
# ===============================

# We now insert a wall 
robot_pos2 = np.array([5.0, 0.0])
wall_start = np.array([10.0, -8.0])          # wall segment
wall_end   = np.array([10.0,  8.0])
robot_pts2 = []
wall_pts   = []

for angle in angles:
    # check robot first
    pt_r = check_circle_intersection(angle, lidar_pos, robot_pos2, robot_radius)
    if pt_r is not None:    # add noise
        xn, yn = add_noise(pt_r[0], pt_r[1], lidar_pos)
        robot_pts2.append((xn, yn))
        continue    # robot occludes wall behind it

    # if no robot hit, check wall as wall is behind the robot
    pt_w = check_wall_intersection(angle, lidar_pos, wall_start, wall_end)
    if pt_w is not None:    # add noise
        xn, yn = add_noise(pt_w[0], pt_w[1], lidar_pos)
        wall_pts.append((xn, yn))

robot_pts2 = np.array(robot_pts2)
wall_pts   = np.array(wall_pts)

fig, ax = plt.subplots(figsize=(8, 7))
ax.add_patch(patches.Circle((robot_pos2), robot_radius, color='steelblue',
             fill=False, lw=2, label='Cylindrical robot'))
ax.plot([wall_start[0], wall_end[0]],
        [wall_start[1], wall_end[1]],
        color='gray', lw=3, label='Wall')
ax.scatter(robot_pts2[:,0], robot_pts2[:,1], s=10,
           color='red',   label='Robot point cloud')
ax.scatter(wall_pts[:,0],   wall_pts[:,1],   s=10,
           color='orange', label='Wall point cloud')
ax.scatter(*lidar_pos, color='blue', s=80,
           marker='^', label='LiDAR origin', zorder=5)
ax.set_aspect('equal'); ax.grid(); ax.legend()
ax.set_title('Exercise 5 --- Robot in front of wall')
plt.tight_layout()
plt.show()

# ===============================
# Exercise 6: Cluster wall + line fitting (Ex. 2 optimization)
# ===============================
 
def opt(params, x, y):
    """Minimize sum of squared distances from points to line ax+by+c=0."""
    a, b, c = params
    # (a, b) = normal vector, defines orientation/direction perpendicular to the line
    # c      = offset from origin, how far the line is shifted from (0,0)
    # numerator:   (ax+by+c)^2 = squared perpendicular distance from each point to line
    # denominator: (a^2+b^2)   = normalises for length of normal vector
    return np.sum((a*x + b*y + c)**2 / (a**2 + b**2))

# using all rays which hit the wall
x_w = wall_pts[:, 0]
y_w = wall_pts[:, 1]

# initial guess: vertical line at wall_x
initial_guess = [1.0, 0.0, -wall_start[0]]
# we minimized based of our cost function here sum of squared distances from points to line 
result  = minimize(opt, initial_guess, args=(x_w, y_w))
# we get best parameters which dsribe it
a_f, b_f, c_f = result.x

# reconstruct fitted line for plotting
y_plot = np.array([y_w.min(), y_w.max()])
x_plot = -(b_f*y_plot + c_f) / a_f

fig, ax = plt.subplots(figsize=(8, 7))
ax.add_patch(patches.Circle((robot_pos2), robot_radius, color='steelblue',
             fill=False, lw=2, label='Cylindrical robot'))
ax.plot([wall_start[0], wall_end[0]],
        [wall_start[1], wall_end[1]],
        color='gray', lw=3, alpha=0.4, label='True wall')
ax.scatter(robot_pts2[:,0], robot_pts2[:,1], s=10,
           color='red',   alpha=0.4, label='Robot points')
ax.scatter(x_w, y_w, s=12, color='green', label='Wall cluster')
ax.plot(x_plot, y_plot, 'k--', lw=2,
        label=f'Fitted: {a_f:.2f}x + {b_f:.2f}y + {c_f:.2f} = 0')
ax.scatter(*lidar_pos, color='blue', s=80,
           marker='^', label='LiDAR origin', zorder=5)
ax.set_aspect('equal'); ax.grid(); ax.legend()
ax.set_title('Exercise 6 --- Wall cluster + line fitting')
plt.tight_layout()
plt.show()

print(f"Fitted line: {a_f:.4f}x + {b_f:.4f}y + {c_f:.4f} = 0")
print(f"True wall at x = {wall_start[0]}")