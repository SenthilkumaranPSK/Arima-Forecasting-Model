import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Import the package components
# Note: In a real project, you would install the package first: pip install -e .
from arima_forecaster import ARIMAForecaster

def generate_sample_data(n_points: int = 100) -> pd.DataFrame:
    """Generates a sample DataFrame with a trend and noise."""
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(n_points)]
    # Linear trend + sine wave + random noise
    values = [10 + (0.5 * i) + (5 * np.sin(i * 0.2)) + np.random.normal(0, 2) for i in range(n_points)]
    
    return pd.DataFrame({
        "Date": dates,
        "Sales": values
    })

def main():
    print("--- ARIMA Forecasting Module Demo ---")

    # 1. Load/Generate Sample Data
    print("\n1. Generating sample time series data...")
    df = generate_sample_data(120)
    print(df.head())

    # 2. Initialize the Forecaster
    # This class handles orchestration of preprocessing, tuning, and fitting.
    forecaster = ARIMAForecaster()

    # 3. Prepare the Data
    # Converts DataFrame columns to a sorted, cleaned pandas Series with a datetime index.
    print("\n2. Preparing data for analysis...")
    forecaster.prepare_data(df, date_col='Date', target_col='Sales', freq='D')

    # 4. Automatic Tuning and Fitting
    # This step:
    # - Performs ADF test to find the optimal differencing parameter 'd'.
    # - Grid searches p and q values to minimize AIC.
    # - Fits the final model.
    print("\n3. Starting automatic model selection and fitting...")
    forecaster.auto_fit(p_range=[0, 1, 2], q_range=[0, 1, 2], max_d=2)

    # 5. Run Diagnostics
    # Validates that residuals are white noise (Ljung-Box test).
    print("\n4. Running model diagnostics...")
    forecaster.run_diagnostics()

    # 6. Generate Forecasts
    # Forecasts 14 days into the future with 95% confidence intervals.
    print("\n5. Generating 14-day forecast...")
    forecast_df = forecaster.forecast(steps=14)
    
    print("\n--- Forecast Results (Next 5 Steps) ---")
    print(forecast_df.head())

    # 7. Visualize Results
    # Plots historical data vs. predicted values.
    print("\n6. Visualizing results...")
    forecaster.plot_results(forecast_steps=14)

if __name__ == "__main__":
    main()
