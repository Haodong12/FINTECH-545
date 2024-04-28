import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.stats as stats

# Load the results from the CSV file containing the returns and volatility comparison
results_df = pd.read_csv('returns_comparison_with_volatility.csv', index_col='Date', parse_dates=True)

# Load individual predictions from the CSV file
individual_tree_predictions_df = pd.read_csv('individual_tree_predictions.csv')


# Density Plot for Returns
plt.figure(figsize=(10, 6))
sns.kdeplot(results_df['Actual Returns'], label='Actual Returns', fill=True)
sns.kdeplot(results_df['RF Returns'], label='RF Predicted Returns', fill=True)
sns.kdeplot(results_df['LR Returns'], label='LR Predicted Returns', fill=True)
plt.title('Predicted vs Actual Returns Distribution')
plt.xlabel('Returns')
plt.ylabel('Density')
plt.legend()
plt.savefig('density_plot.png')  # Save the plot as a PNG file
plt.show()

# Residual Plot for Random Forest Model
plt.figure(figsize=(10, 6))
rf_residuals = results_df['Actual Returns'] - results_df['RF Returns']
sns.scatterplot(x=results_df['Actual Returns'], y=rf_residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residual Plot for RF Model Predictions')
plt.xlabel('Actual Returns')
plt.ylabel('Residuals')
plt.savefig('rf_residual_plot.png')
plt.show()

# Residual Plot for Linear Regression Model
plt.figure(figsize=(10, 6))
lr_residuals = results_df['Actual Returns'] - results_df['LR Returns']
sns.scatterplot(x=results_df['Actual Returns'], y=lr_residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residual Plot for LR Model Predictions')
plt.xlabel('Actual Returns')
plt.ylabel('Residuals')
plt.savefig('lr_residual_plot.png')
plt.show()

# Boxplot for the distribution of predictions
for observation_index in range(5):  
    plt.figure(figsize=(10, 6))
    # Make sure you are using 'individual_tree_predictions_df' instead of 'individual_predictions_df'
    plt.boxplot(individual_tree_predictions_df.iloc[observation_index, :].dropna(), vert=False)
    plt.title(f'Distribution of Predictions for Observation {observation_index}')
    plt.xlabel('Predicted Value')
    # Use 'get_yaxis().set_visible(False)' to hide y-axis ticks if needed
    plt.gca().get_yaxis().set_visible(False)
    plt.savefig(f'individual_predictions_boxplot_{observation_index}.png')
    plt.show()

# Line Plot for Actual vs Predicted Values
plt.figure(figsize=(14, 7))
plt.plot(results_df.index, results_df['Actual Returns'], label='Actual Values', color='red', marker='')
plt.plot(results_df.index, results_df['RF Returns'], label='RF Predicted Values', color='blue', marker='')
plt.plot(results_df.index, results_df['LR Returns'], label='LR Predicted Values', color='green', marker='')
plt.title('Actual vs Predicted Values')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.savefig('Actual vs Predicted Values.png')
plt.show()
