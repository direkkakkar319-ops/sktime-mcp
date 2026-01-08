from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.compose import TransformedTargetForecaster
from sktime.transformations.series.boxcox import LogTransformer

# Create the pipeline components
step_0 = LogTransformer()
step_1 = AutoARIMA()

# Define the pipeline
pipeline = TransformedTargetForecaster([
    ("LogTransformer", step_0),
    ("AutoARIMA", step_1),
])

# Example usage:
# Load data
from sktime.datasets import load_airline
y = load_airline()

# Fit the model
pipeline.fit(y)

# Make predictions
fh = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # 12-step ahead forecast
predictions = pipeline.predict(fh=fh)
print(predictions)
