import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

phi = 3e-3
beta_sq = np.tan(phi)**2
Dr = 0.15
eta_sys = 0.9
scattering_eff = 0.17
rho = 0.2
backscatter = 0.5
eta_atm = 0.9
R = 150


def opt(params, x, y):
    a, b, c = params
    numerator = (a*x + b*y + c)**2
    denominator = a**2 + b**2
    return np.sum(numerator/denominator) 

def line_circle_intersection_abc(a, b, c, h, k, r):

    # Find closest point from circle center to the line
    denom = a*a + b*b
    x0 = (b*(b*h - a*k) - a*c) / denom
    y0 = (a*(-b*h + a*k) - b*c) / denom

    # Distance from center to line
    d = abs(a*h + b*k + c) / np.sqrt(denom)

    if d > r:
        return None  # no intersection

    if abs(d - r) < 1e-9:
        return (x0, y0) # tangent (1 point)

    # distance from closest point to intersections
    mult = np.sqrt((r*r - d*d) / denom)

    x1 = x0 + b * mult
    y1 = y0 - a * mult

    x2 = x0 - b * mult
    y2 = y0 + a * mult

    # choose closest point
    dist1 = np.sqrt(x1**2 + y1**2)
    dist2 = np.sqrt(x2**2 + y2**2)

    if dist1 < dist2:
        return (x1,y1)
    else:
        return (x2, y2)

def ray_circle_intersections(h, k, r, angles):
    """
    Compute intersection points of rays from (0,0) to a circle.

    Parameters:
    h, k : float
    Circle center
    r : float
    Circle radius
    angles : np.ndarray
    Array of angles in radians

    Returns:
    points : list of tuples
    Each element is [(x1,y1),(x2,y2)] for that ray
    """
    points = []
    for theta in angles:
    # line coefficients for ray from origin
        a = np.sin(theta)
        b = -np.cos(theta)
        c = 0

    # call your line-circle intersection function
        intersection = line_circle_intersection_abc(a, b, c, h, k, r)
        points.append(intersection)

    return points

if __name__ == "__main__":
    
    power_ratio = (4/(np.pi * R**2 * beta_sq))*rho*scattering_eff*(1/(backscatter*R**2))*((np.pi*Dr**2)/4)*eta_atm*eta_sys


    ## initial_guess = [1, 1, 0]
    ## result = minimize(opt, initial_guess, args=(x, y))
    ## a, b, c = result.x

    # circle at (2,3) with radius 5
    h, k, r = 10, 10, 5
    N = 100

    # 10 rays from origin
    angles = np.linspace(0, 0.5*np.pi, N, endpoint=False)

    points = ray_circle_intersections(h, k, r, angles)

    # Flatten the points into separate x and y lists
    x_list = []
    y_list = []


    for pt in points:
        if pt is not None:
            x_list.append(pt[0])
            y_list.append(pt[1])
    # Convert to numpy arrays
    x_arr = np.array(x_list)
    y_arr = np.array(y_list)

    noise_std = 0.05

    samples= len(y_arr)
    x_noisy = x_arr + np.random.normal(0, noise_std, samples)
    y_noisy = y_arr + np.random.normal(0, noise_std, samples)

    # Scatter plot
    plt.scatter(x_noisy, y_noisy, color='red')
    plt.scatter(0, 0, color='blue')
    plt.gca().set_aspect('equal', adjustable='box')

    # Optional: plot the circle for reference
    circle = plt.Circle((h,k), r, color='blue', fill=False)
    plt.gca().add_artist(circle)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Ray-Circle Intersections')
    plt.show()