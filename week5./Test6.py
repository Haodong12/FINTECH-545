import pandas as pd
import numpy as np

# Function to read stock prices and calculate returns
def calculate_returns(file_path):
    # Read the CSV file. Assuming the first row and first column are to be used as headers and index, respectively
    df = pd.read_csv(file_path, index_col='Date', parse_dates=True)

    # Calculate arithmetic returns
    arithmetic_returns = df.pct_change()
    arithmetic_returns.to_csv('test6_1.csv')
    # Calculate log returns
    log_returns = np.log(df / df.shift(1))
    arithmetic_returns.to_csv('test6_2.csv')
    return arithmetic_returns, log_returns


file_path = '/Users/Lenovo/Desktop/Week5/test_files/test6.csv'  
arithmetic_returns, log_returns = calculate_returns(file_path)

print("Arithmetic Returns:")
print(arithmetic_returns)

print("\nLogarithmic Returns:")
print(log_returns)
