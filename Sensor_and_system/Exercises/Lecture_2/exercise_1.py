import numpy as np

TRUE_MEAN = 2.0
TRUE_VAR  = 67 / 50   # 1.34
N = 2000

# To sample from a PDF we need the inverse CDF (F^-1):
# draw u ~ U(0,1) and map it to x = F^-1(u)

# The CDF is built by integrating f(x) piece by piece, each piece
# starting where the previous one ended (boundary carries over).
# Then we solve for x to get the inverse.

# Piece 2 example:
# F(x) = 3/10 + [(x-2)^3 + 1]/5   (3/10 = F(1), carried from piece 1)
# Solve u = F(x) for x  =>  x = 2 + cbrt(5u - 5/2)
def sample_icdf(u):
    u = np.asarray(u, dtype=float)
    x = np.empty_like(u)
 
    # Piece 1:
    m1 = u <= 3/10                           #F(0) = 0/10 and F(1) = 3/10
    x[m1] = np.sqrt(10 * u[m1] / 3)
    #  F(x) = 3x^2/10  =>  F^-1(u) = sqrt(10u/3)

    # Piece 2:
    m2 = (u > 3/10) & (u <= 7/10)           #F(1) = 3/10 and F(3) = 7/10
    x[m2] = 2 + np.cbrt(5 * u[m2] - 5/2)
    #F(x) = 3/10 + [(x-2)^3 + 1]/5  =>  F^-1(u) = 2 + cbrt(5u - 5/2)

    # Piece 3:
    m3 = u > 7/10                               #F(3) = 7/10 and F(4) = 10/10
    x[m3] = 4 - np.sqrt(120 * (1 - u[m3])) / 6  
    #F(x) = [24x - 3x^2 - 38]/10  =>  F^-1(u) = 4 - sqrt(120(1-u))/6
 
    return x
 
# ── Run ────────────────────────────────────────────────────────────────────────
 
samples = sample_icdf(np.random.uniform(0, 1, N))
 
sample_mean = samples.mean()
sample_var  = samples.var(ddof=1)   # divide by N-1 (unbiased estimator)
 
print(f"Mean Sample {sample_mean:.4f} and True {TRUE_MEAN:.4f}")
print(f"Variance Sample {sample_var:.4f} and True {TRUE_VAR:.4f}")
# more smaple should result in close