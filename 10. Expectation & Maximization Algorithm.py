import numpy as np
from scipy.stats import norm
np.random.seed(42)
data = np.concatenate([
    np.random.normal(0, 1, 100),
    np.random.normal(5, 1, 100)
])

# Initialize parameters
mu1, mu2 = np.random.rand(2) * 10
sigma1, sigma2 = np.random.rand(2) * 5 + 1e-6  # avoid zero variance
pi = 0.5
for _ in range(100):
    # Expectation step
    likelihood1 = norm.pdf(data, mu1, sigma1)
    likelihood2 = norm.pdf(data, mu2, sigma2)

    weight1 = (pi * likelihood1) / (
        pi * likelihood1 + (1 - pi) * likelihood2
    )
    weight2 = 1 - weight1
    mu1 = np.sum(weight1 * data) / np.sum(weight1)
    mu2 = np.sum(weight2 * data) / np.sum(weight2)

    sigma1 = np.sqrt(
        np.sum(weight1 * (data - mu1) ** 2) / np.sum(weight1)
    )
    sigma2 = np.sqrt(
        np.sum(weight2 * (data - mu2) ** 2) / np.sum(weight2)
    )

    pi = np.mean(weight1)

print("Final Parameters:")
print(f"Cluster 1 - Mean: {mu1:.2f}, Standard Deviation: {sigma1:.2f}")
print(f"Cluster 2 - Mean: {mu2:.2f}, Standard Deviation: {sigma2:.2f}")
print(f"Cluster Weights - Cluster 1: {pi:.2f}, Cluster 2: {1 - pi:.2f}")
