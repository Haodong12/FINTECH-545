# fit the data with OLS and MLE, then compare beta coefficients and std of the OLS error to the std of MLE error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def parse_csv(file_name):
    # Read the CSV file into a DataFrame
    data = pd.read_csv(file_name)
    return data

df = parse_csv('problem2.csv')  

df['intercept'] = 1 # Add a column of ones for the intercept X0
X = df[['intercept', 'x']].values  # Independent variable
y = df['y'].values  # Dependent variable

# Calculate the OLS estimate
# beta = (X'X)^(-1)X'y
beta_ols = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

# Calculate residuals and standard deviation of errors
residuals = y - X.dot(beta_ols)
n = len(y)
p = X.shape[1] # number of coluns for parameters
residual_variance = residuals.T.dot(residuals) / (n - p)

# Calculate the standard errors 
XX_inv = np.linalg.inv(X.T.dot(X))
std_errors = np.sqrt(np.diag(XX_inv) * residual_variance)

# Results
print("OLS Betas:", beta_ols[1])
print("OLS Standard Errors:", std_errors) # intercept and slope

# Plotting the OLS fit
plt.figure(figsize=(10, 6))
plt.scatter(df['x'], y, color='blue', label='Data')
plt.plot(df['x'], X.dot(beta_ols), color='red', label='OLS Fit')

plt.title('OLS Regression Fit')
plt.xlabel('Independent variable (x)')
plt.ylabel('Dependent variable (y)')
plt.legend()
plt.show(block = False)

# MLE estimate
# negative log-likelihood
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

# Plot MLE fit
y_mle_pred = X.dot(mle_betas)

plt.figure(figsize=(10, 6))
plt.scatter(df['x'], y, color='blue', label='Data')
plt.plot(df['x'], y_mle_pred, color='red', label='MLE Fit')
plt.title('MLE Regression Fit')
plt.xlabel('Independent variable (x)')
plt.ylabel('Dependent variable (y)')
plt.legend()
plt.show()