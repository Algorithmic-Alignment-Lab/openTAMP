import numpy as np

# samples a brownian motion of a given drift and volatility, given sampling at timesteps 
def brownian_motion(timesteps, drift, vol):
    increments = np.concatenate([[0],np.ediff1d(timesteps)])
    means = drift * np.ones(len(timesteps))
    cov = np.diag(vol * increments)  # variances of Gaussians
    gaus_sample = np.random.multivariate_normal(means, cov)
    # print(gaus_sample)
    return np.cumsum(gaus_sample)

# print(np.ediff1d([0, 1]))
print(brownian_motion(np.array([0, 1, 2, 3, 8]), 10, 1000))
