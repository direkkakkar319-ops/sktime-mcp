from sktime.forecasting.naive._naive import NaiveForecaster
from sktime.datasets import load_airline

# 1. Instantiate the estimator
# This matches the configuration used in the MCP session
forecaster = NaiveForecaster(strategy='last', window_length=None, sp=1)

# 2. Load the dataset
# usage: predicting on the airline dataset
y = load_airline()

# 3. Fit the model
forecaster.fit(y)

# 4. Make predictions
# Predict the next time step (Horizon = 1)
fh = [1]
predictions = forecaster.predict(fh=fh)

print("Forecast for the next step:")
print(predictions)
