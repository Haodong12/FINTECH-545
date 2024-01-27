# MLE with T-Distributed Errors
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import t

def parse_csv(file_name):
    # Read the CSV file into a DataFrame
    data = pd.read_csv(file_name)
    return data

df = parse_csv('problem2.csv')  

df['intercept'] = 1 # Add a column of ones for the intercept X0
X = df[['intercept', 'x']].values  # Independent variable
y = df['y'].values  # Dependent variable
def neg_log_likelihood(params, X, y):
    n = len(y)
    betas = params[:-1]
    sigma = params[-1]

    # Calculate the predicted values from the linear model
    y_pred = X.dot(betas)

    # Calculate the residuals
    residuals = y - y_pred

    # Compute the negative log-likelihood
    neg_ll = (n/2) * np.log(2 * np.pi) + (n/2) * np.log(sigma**2) + (1/(2 * sigma**2)) * np.sum(residuals**2)
    return neg_ll

# Initial parameters for MLE
initial_params = np.zeros(X.shape[1] +1 )
initial_params[-1] = 1 # 0 for betas, 1 for sigma

# Optimization to find the MLE estimates
result = minimize(neg_log_likelihood, initial_params, args=(X, y))

# Print the results
if result.success:
    mle_betas = result.x[:-1]
    mle_sigma = result.x[-1]
    print('MLE Betas:', mle_betas)
    print('MLE Standard deivation of residuals:', mle_sigma)
else:
    raise ValueError('MLE optimization failed:', result.message)

# Negative log-likelihood for T-distribution errors
def neg_log_likelihood_t(params, X, y):
    betas = params[:-2]
    sigma = params[-2]  # Use exp to ensure sigma is positive
    log_dof = params[-1]  # Use exp to ensure dof is positive
    dof = np.exp(log_dof)   # Degrees of freedom for the T-distribution

    # Calculate the predicted values from the linear model
    y_pred = X.dot(betas)

    # Calculate the residuals
    residuals = y - y_pred

    # Compute the negative log-likelihood for T-distributed errors
    neg_ll = -np.sum(t.logpdf(residuals, dof, scale=sigma))
    return neg_ll

# Initial parameters for MLE with T-distribution
initial_params_t = np.append(np.zeros(X.shape[1]), [1,10])  # Last two are sigma and dof

# Optimization to find the MLE estimates assuming T-distributed errors
result_t = minimize(neg_log_likelihood_t, initial_params_t, args=(X, y))

# Print the results
if result_t.success:
    mle_betas_t = result_t.x[:-2]
    mle_sigma_t = result_t.x[-2]
    mle_df_t = result_t.x[-1]
    print('MLE Betas (T-distribution):', mle_betas_t)
    print('MLE Sigma (T-distribution):', mle_sigma_t)
    print('MLE Degrees of Freedom (T-distribution):', mle_df_t)
else:
    raise ValueError('MLE optimization failed for T-distribution:', result_t.message)

# Function to calculate R-squared and Adjusted R-squared
p = 1
p_t = p + 1

def calc_r_squared(y, y_pred, p):
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    n = len(y)
    adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)
    return r_squared, adj_r_squared

# Calculate R-squared and Adjusted R-squared for both models
r_squared_normal, adj_r_squared_normal = calc_r_squared(y, X.dot(mle_betas), p)
r_squared_t, adj_r_squared_t = calc_r_squared(y, X.dot(mle_betas_t), p + 1)  # +1 for dof

# Calculate Log-Likelihood at the estimated parameters
params_normal = np.append(mle_betas, mle_sigma)
n = len(y)
log_likelihood_normal = -neg_log_likelihood(params_normal, X, y)
log_likelihood_t = -neg_log_likelihood_t(result_t.x, X, y)

# Calculate AIC and BIC
aic_normal = 2 * (p + 1) - 2 * log_likelihood_normal  # +1 for sigma
bic_normal = np.log(n) * (p + 1) - 2 * log_likelihood_normal

aic_t = 2 * (p + 2) - 2 * log_likelihood_t  # +2 for sigma and dof
bic_t = np.log(n) * (p + 2) - 2 * log_likelihood_t

# Print results
print("Adjusted R-squared (Normal):", adj_r_squared_normal)
print("AIC (Normal):", aic_normal)
print("BIC (Normal):", bic_normal)

print("Adjusted R-squared (T-Distribution):", adj_r_squared_t)
print("AIC (T-Distribution):", aic_t)
print("BIC (T-Distribution):", bic_t)