import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def parse_csv(file_name):
    # Read the CSV file into a DataFrame
    data = pd.read_csv(file_name)
    return data

df_x = parse_csv('problem2_x.csv')  
df_x1 = parse_csv('problem2_x1.csv')  

mean = df_x.mean()
covariance = df_x.cov()

# Extract the necessary values from the mean and covariance
mu_x1, mu_x2 = mean['x1'], mean['x2']
var_x1, var_x2 = covariance.loc['x1', 'x1'], covariance.loc['x2', 'x2']
cov_x1_x2 = covariance.loc['x1', 'x2']

# Conditional mean and variance calculations
x1_observed = df_x1['x1']
conditional_mean_x2 = mu_x2 + (cov_x1_x2 / var_x1) * (x1_observed - mu_x1)
conditional_variance_x2 = var_x2 - (cov_x1_x2**2 / var_x1)

# 95% confidence interval calculations
z_score_95 = 1.96  # Z-score for 95% confidence
ci_lower = conditional_mean_x2 - z_score_95 * np.sqrt(conditional_variance_x2)
ci_upper = conditional_mean_x2 + z_score_95 * np.sqrt(conditional_variance_x2)

# Plotting the expected value and confidence interval
plt.figure(figsize=(10, 6))
plt.plot(x1_observed, conditional_mean_x2, label='Conditional Mean of X2', color='blue')
plt.fill_between(x1_observed, ci_lower, ci_upper, color='gray', alpha=0.5, label='95% CI')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Conditional Distribution of X2 given X1 with 95% Confidence Interval')
plt.legend()
plt.show()