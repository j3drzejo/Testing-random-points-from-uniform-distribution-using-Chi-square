import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

def chi_square_test(observed, expected):
    return np.sum((observed - expected)**2 / expected)

def test_uniform_distribution(random_numbers, num_bins):
    observed, _ = np.histogram(random_numbers, bins=num_bins)
    expected = len(random_numbers) / num_bins
    chi_square = chi_square_test(observed, expected)
    df = num_bins - 1
    critical_value = chi2.ppf(0.99, df)  # Significance level of 0.01
    return chi_square, critical_value, observed, expected

# Generate random numbers
random_numbers = np.random.rand(10000)

# Number of bins
num_bins = 10000

# Perform the test
chi_square, critical_value, observed, expected = test_uniform_distribution(random_numbers, num_bins)

# Plot histogram
plt.hist(random_numbers, bins=num_bins, edgecolor='black')
plt.title('Histogram of Random Numbers')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()

# Print results
print("Chi-square statistic:", chi_square)
print("Critical value:", critical_value)
if chi_square <= critical_value:
    print("The PRNG produces random numbers consistent with a uniform distribution.")
else:
    print("The PRNG does not produce random numbers consistent with a uniform distribution.")
