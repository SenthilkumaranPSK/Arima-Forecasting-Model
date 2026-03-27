import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from arima_forecaster.preprocessing import prepare_time_series, check_stationarity_detailed
from arima_forecaster.tuning import find_optimal_params
from arima_forecaster.core import ARIMAForecaster

@pytest.fixture
def sample_df():
    """Generates a sample DataFrame with a trend and some noise."""
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(100)]
    values = [10 + (0.5 * i) + np.random.normal(0, 1) for i in range(100)]
    return pd.DataFrame({"Date": dates, "Sales": values})

def test_prepare_time_series(sample_df):
    """Verifies that preprocessing returns a cleaned Series with a DatetimeIndex."""
    series = prepare_time_series(sample_df, 'Date', 'Sales', freq='D')
    
    assert isinstance(series, pd.Series)
    assert isinstance(series.index, pd.DatetimeIndex)
    assert series.index.freqstr == 'D'
    assert not series.isnull().any()
    assert len(series) == 100

def test_stationarity_check():
    """Verifies stationarity detection and automatic differencing."""
    # Create a non-stationary random walk
    np.random.seed(42)
    non_stationary = pd.Series(np.cumsum(np.random.normal(0, 1, 100)))
    
    # Create a stationary white noise
    stationary = pd.Series(np.random.normal(0, 1, 100))
    
    # 1. Test stationary data
    _, res_stat = check_stationarity_detailed(stationary)
    assert res_stat.is_stationary == True
    assert res_stat.d == 0
    
    # 2. Test non-stationary data (should require at least d=1)
    _, res_non_stat = check_stationarity_detailed(non_stationary)
    assert res_non_stat.d > 0

def test_tuning_format(sample_df):
    """Verifies that the tuning engine returns the correct format and data types."""
    series = prepare_time_series(sample_df, 'Date', 'Sales', freq='D')
    # We use a very small range for speed in testing
    best_order, summary = find_optimal_params(series, p_range=[0, 1], d_range=[1], q_range=[0, 1])
    
    assert isinstance(best_order, tuple)
    assert len(best_order) == 3
    assert isinstance(summary, pd.DataFrame)
    assert 'order' in summary.columns
    assert 'aic' in summary.columns

def test_core_forecaster_flow(sample_df):
    """Tests the end-to-end integration of the ARIMAForecaster class."""
    forecaster = ARIMAForecaster()
    
    # 1. Prepare data
    forecaster.prepare_data(sample_df, 'Date', 'Sales', freq='D')
    assert forecaster.series is not None
    
    # 2. Auto fit (uses small ranges for test speed)
    forecaster.auto_fit(p_range=[0, 1], q_range=[0, 1], max_d=1)
    assert forecaster.results is not None
    assert forecaster.order is not None
    
    # 3. Forecast
    steps = 5
    forecast = forecaster.forecast(steps=steps)
    assert isinstance(forecast, pd.DataFrame)
    assert len(forecast) == steps
    assert all(col in forecast.columns for col in ['mean', 'lower_ci', 'upper_ci'])

def test_prepare_data_error_handling():
    """Verifies that improper data input raises a ValueError."""
    forecaster = ARIMAForecaster()
    df_missing_cols = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    
    with pytest.raises(ValueError, match="not found in DataFrame"):
        forecaster.prepare_data(df_missing_cols, 'Date', 'Sales')
