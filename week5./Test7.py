import pandas as pd
from scipy import stats
import statsmodels.api as sm

def fit_normal_distribution(file_path):
    # Read the CSV file
    data = pd.read_csv(file_path)
    x = data['x1']

    # Fit a normal distribution
    mu, std = stats.norm.fit(x)

    # Save the parameters to a CSV file
    pd.DataFrame({'Parameter': ['mu', 'std'], 'Value': [mu, std]}).to_csv('testout_7.1.csv', index=False)

def fit_t_distribution(file_path):
    # Read the CSV file
    data = pd.read_csv(file_path)
    x = data['x1']

    # Fit a t-distribution
    df, loc, scale = stats.t.fit(x)

    # Save the parameters to a CSV file
    pd.DataFrame({'Parameter': ['df', 'loc', 'scale'], 'Value': [df, loc, scale]}).to_csv('testout_7.2.csv', index=False)

def Regression(file_path3):
    # Read the CSV file
    data = pd.read_csv(file_path)
    X = data[['x1', 'x2', 'x3']]
    y = data['y']

    # Adding a constant to the model (intercept)
    X = sm.add_constant(X)

    # Fit a linear model
    model = sm.OLS(y, X).fit()

    # Now fit a robust linear model using Huber's T-regression approach
    robust_model = sm.RLM(y, X, M=sm.robust.norms.HuberT()).fit()

    # Prepare a DataFrame to save results
    results_df = pd.DataFrame({
        'Parameter': ['const', 'x1', 'x2', 'x3'],
        'OLS_Coefficients': model.params,
        'Robust_Coefficients': robust_model.params,
    })

    # Save the parameters to a CSV file
    results_df.to_csv('testout_7.3.csv', index=False)


file_path_1 = '/Users/Lenovo/Desktop/Week5/test_files/test7_1.csv'   
fit_normal_distribution(file_path_1)

file_path_2 = '/Users/Lenovo/Desktop/Week5/test_files/test7_2.csv'   
fit_t_distribution(file_path_2)

file_path = '/Users/Lenovo/Desktop/Week5/test_files/test7_3.csv'   
Regression(file_path)
