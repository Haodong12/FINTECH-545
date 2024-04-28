import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Load the prepared data
data = pd.read_csv('prepared_stock_data.csv', index_col='Date', parse_dates=True)

# Function to calculate returns
def calculate_returns(prices):
    return prices.pct_change()

# Define features used for training, now including MA20 and MA50, and excluding RSI
features = ['Adj Close', 'Open', 'High', 'Low', 'MA5', 'MA10', 'MA20', 'MA50', 'nasdaq']
X = data[features]
y = data['Target']

# Split data into training and testing sets based on the date
X_train = data.loc['2012-01-01':'2022-12-31', features]
y_train = data.loc['2012-01-01':'2022-12-31', 'Target']
X_test = data.loc['2023-01-01':'2023-12-31', features]
y_test = data.loc['2023-01-01':'2023-12-31', 'Target']

# Initialize and train the models
rf = RandomForestRegressor(n_estimators=1000, max_depth=30, min_samples_split=20, min_samples_leaf=20, random_state=1)
lr = LinearRegression()
rf.fit(X_train, y_train)
lr.fit(X_train, y_train)

# Save the trained Random Forest model
joblib.dump(rf, 'random_forest_model.joblib')

# Get individual tree predictions
individual_tree_predictions = np.array([tree.predict(X_test) for tree in rf.estimators_])
volatility_per_instance = individual_tree_predictions.std(axis=0)
individual_predictions_df = pd.DataFrame(individual_tree_predictions.T)
individual_predictions_df.to_csv('individual_tree_predictions.csv')

# Calculate the mean prediction across all trees (point estimate) and linear regression predictions
rf_predictions = individual_tree_predictions.mean(axis=0)
lr_predictions = lr.predict(X_test)

# Calculate returns and volatilities
actual_returns = calculate_returns(y_test)
rf_predicted_prices = pd.Series(rf_predictions, index=y_test.index)
lr_predicted_prices = pd.Series(lr_predictions, index=y_test.index)
rf_returns = calculate_returns(rf_predicted_prices)
lr_returns = calculate_returns(lr_predicted_prices)

actual_volatility = actual_returns.std()
lr_volatility = lr_returns.std()

# Save returns and volatility to CSV
results_df = pd.DataFrame({
    "Actual Returns": actual_returns,
    "RF Returns": rf_returns,
    "LR Returns": lr_returns,
    "RF Volatility": volatility_per_instance,
    "LR Volatility": lr_volatility,
    "Actual Volatility": actual_volatility,
})
results_df.to_csv("returns_comparison_with_volatility.csv")

# Calculate and print metrics
metrics = {
    "RF_RMSE": np.sqrt(mean_squared_error(y_test, rf_predictions)),
    "LR_RMSE": np.sqrt(mean_squared_error(y_test, lr_predictions)),
    "RF_MAE": mean_absolute_error(y_test, rf_predictions),
    "LR_MAE": mean_absolute_error(y_test, lr_predictions),
    "RF_R2": r2_score(y_test, rf_predictions),
    "LR_R2": r2_score(y_test, lr_predictions),
    "RF Volatility": np.mean(volatility_per_instance),
    "LR Volatility": lr_volatility,
    "Actual Volatility": actual_volatility
}

for key, value in metrics.items():
    print(f"{key}: {value}")

# Display feature importances
importances = rf.feature_importances_
feature_names = features
feature_importance_dict = dict(zip(feature_names, importances))
print("Feature importances:", feature_importance_dict)
