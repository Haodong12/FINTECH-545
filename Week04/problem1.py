import numpy as np

# Set the random seed for reproducibility
np.random.seed(1)

# Parameters for the simulation with updated sigma
num_simulations = 10000  # Number of simulations to perform
sigma = 0.5  # standard deviation for the normal distribution of returns
initial_price = 100  # Example initial price P_t-1

# Generate random returns from normal distribution with updated sigma
random_returns = np.random.normal(0, sigma, num_simulations)

# Calculate prices using the three different return equations
prices_classical = initial_price + random_returns
prices_arithmetic = initial_price * (1 + random_returns)
prices_log = initial_price * np.exp(random_returns)

# Calculate the expected values (means) and standard deviations
mean_classical = np.mean(prices_classical)
std_dev_classical = np.std(prices_classical)

mean_arithmetic = np.mean(prices_arithmetic)
std_dev_arithmetic = np.std(prices_arithmetic)

mean_log = np.mean(prices_log)
std_dev_log = np.std(prices_log)

# Print the results
print(f"Classical Brownian Motion:\n Mean: {mean_classical}, Standard Deviation: {std_dev_classical}")
print(f"Arithmetic Return System:\n Mean: {mean_arithmetic}, Standard Deviation: {std_dev_arithmetic}")
print(f"Log Return or Geometric Brownian Motion:\n Mean: {mean_log}, Standard Deviation: {std_dev_log}")