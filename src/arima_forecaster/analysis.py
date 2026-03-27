import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Any, cast
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.diagnostic import acorr_ljungbox

def check_stationarity(series: pd.Series) -> dict:
    """
    Performs the Augmented Dickey-Fuller (ADF) test to check for stationarity.

    Args:
        series: The time series to test.

    Returns:
        dict: A dictionary containing the ADF statistic, p-value, and stationarity status.
    """
    result = cast(Any, adfuller(series.dropna()))
    p_value = float(result[1])
    is_stationary = p_value < 0.05
    
    return {
        "adf_statistic": float(result[0]),
        "p_value": p_value,
        "is_stationary": is_stationary,
        "critical_values": result[4]
    }

def make_stationary(series: pd.Series, max_d: int = 2) -> Tuple[pd.Series, int]:
    """
    Differences the series until it becomes stationary or max_d is reached.

    Args:
        series: The input time series.
        max_d: Maximum number of differencing operations.

    Returns:
        Tuple[pd.Series, int]: The stationary series and the number of differences (d) applied.
    """
    d = 0
    temp_series = series.copy().dropna()
    
    while d < max_d:
        res = check_stationarity(temp_series)
        if res["is_stationary"]:
            break
        d += 1
        temp_series = temp_series.diff().dropna()
        
    return temp_series, d

def plot_lag_analysis(series: pd.Series, lags: int = 40):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4))
    plot_acf(series.dropna(), lags=lags, ax=ax1)
    ax1.set_title("Autocorrelation (ACF)")
    plot_pacf(series.dropna(), lags=lags, ax=ax2, method='ywm')
    ax2.set_title("Partial Autocorrelation (PACF)")
    plt.tight_layout()
    return fig

def analyze_trend(series: pd.Series, model: str = 'additive', period: Optional[int] = None):
    decomposition = seasonal_decompose(series.dropna(), model=model, period=period)
    fig = decomposition.plot()
    fig.set_size_inches(12, 8)
    plt.tight_layout()
    return fig, decomposition

def diagnose_residuals(model_results: Any):
    fig = model_results.plot_diagnostics(figsize=(14, 10))
    lb_test = acorr_ljungbox(model_results.resid, lags=[10], return_df=True)
    return fig, lb_test

def plot_forecast(actual: pd.Series, forecast: pd.Series, conf_int: Optional[pd.DataFrame] = None):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(actual, label="Actual Observations", color='blue')
    ax.plot(forecast, label="Forecast", color='red', linestyle='--')
    if conf_int is not None:
        ax.fill_between(conf_int.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='pink', alpha=0.3)
    ax.legend()
    plt.tight_layout()
    return fig
