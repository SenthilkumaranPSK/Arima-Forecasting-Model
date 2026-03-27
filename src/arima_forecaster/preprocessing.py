import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Union, Any, cast
from statsmodels.tsa.stattools import adfuller

class StationarityResult:
    """
    Container for stationarity test results and metadata.
    
    Attributes:
        is_stationary (bool): True if the series passed the ADF test (p < 0.05).
        p_value (float): The p-value from the Augmented Dickey-Fuller test.
        adf_statistic (float): The calculated ADF statistic.
        critical_values (Dict[str, float]): The critical values for the test.
        d (int): The number of differencing steps applied to reach this state.
    """
    def __init__(self, is_stationary: bool, p_value: float, adf_statistic: float, critical_values: Dict[str, float], d: int):
        self.is_stationary = is_stationary
        self.p_value = p_value
        self.adf_statistic = adf_statistic
        self.critical_values = critical_values
        self.d = d

    def __repr__(self) -> str:
        status = "Stationary" if self.is_stationary else "Non-Stationary"
        return (f"<StationarityResult: {status} (d={self.d}, p={self.p_value:.4f}, "
                f"ADF={self.adf_statistic:.4f})>")

def prepare_time_series(
    df: pd.DataFrame, 
    date_col: str, 
    target_col: str, 
    freq: Optional[str] = None,
    impute_method: str = "linear"
) -> pd.Series:
    """
    Cleans and prepares a pandas DataFrame into a Series suitable for ARIMA.

    Steps:
    1. Validates input columns.
    2. Converts the date column to datetime objects.
    3. Sets the date column as the index and sorts it.
    4. Handles missing values in the target column.
    5. Ensures the index has a consistent frequency (required for statsmodels).

    Args:
        df: The raw input DataFrame.
        date_col: The name of the column containing date/time information.
        target_col: The name of the column containing the values to forecast.
        freq: A pandas frequency string (e.g., 'D' for daily, 'MS' for month start).
              If None, the function will attempt to infer it.
        impute_method: Strategy for handling NaNs ('linear', 'ffill', 'bfill').

    Returns:
        pd.Series: A cleaned Series with a DatetimeIndex and consistent frequency.

    Raises:
        ValueError: If columns are missing or data is insufficient.
    """
    if date_col not in df.columns or target_col not in df.columns:
        raise ValueError(f"Columns '{date_col}' or '{target_col}' not found in DataFrame.")

    # Create a copy to prevent modifying the original DataFrame
    data = df[[date_col, target_col]].copy()
    
    # 1. Date conversion and target conversion with coercion
    # This handles metadata lines by turning them into NaT/NaN
    data[date_col] = pd.to_datetime(data[date_col], errors='coerce')
    data[target_col] = pd.to_numeric(data[target_col], errors='coerce')
    
    # Drop rows where either date or target conversion failed
    data = data.dropna(subset=[date_col, target_col])
    
    # 2. Set index and sort
    data = data.set_index(date_col).sort_index()
    series = data[target_col]
    
    # 3. Handle missing values
    if series.isnull().any():
        if impute_method == "linear":
            series = series.interpolate(method='linear')
        elif impute_method == "ffill":
            series = series.ffill()
        elif impute_method == "bfill":
            series = series.bfill()
            
    # Final cleanup of boundary NaNs
    series = series.dropna()

    # Set and validate frequency (statsmodels ARIMA requires a frequency)
    if freq:
        series = series.asfreq(freq)
    else:
        # Cast index to DatetimeIndex to satisfy static type checkers
        if isinstance(series.index, pd.DatetimeIndex):
            inferred_freq = pd.infer_freq(series.index)
            if inferred_freq:
                series = series.asfreq(inferred_freq)
        else:
            # Fallback for non-datetime indexes if necessary
            pass
            
    return series

def check_stationarity_detailed(series: pd.Series, max_d: int = 2) -> Tuple[pd.Series, StationarityResult]:
    """
    Checks for stationarity and automatically applies differencing if needed.

    Uses the Augmented Dickey-Fuller (ADF) test. If the p-value is >= 0.05, 
    the series is differenced, and the test is repeated up to max_d times.

    Args:
        series: A cleaned pandas Series with a DatetimeIndex.
        max_d: Maximum number of differencing operations allowed (usually 1 or 2).

    Returns:
        Tuple[pd.Series, StationarityResult]: The (potentially) differenced series 
                                               and its statistical metadata.
    """
    d = 0
    current_series = series.copy().dropna()
    
    # Initialize variables to satisfy static analysis
    p_val: float = 1.0
    adf_stat: float = 0.0
    crit_vals: Dict[str, float] = {}
    
    while d <= max_d:
        # Perform ADF Test
        # regression='c' assumes a constant intercept in the series
        # Cast to Any to bypass incorrect type stubs for adfuller return tuple
        result = cast(Any, adfuller(current_series, autolag='AIC'))
        
        adf_stat = float(result[0])
        p_val = float(result[1])
        # result[4] contains the critical values dictionary
        crit_vals = cast(Dict[str, float], result[4])
        is_stationary = p_val < 0.05

        if is_stationary:
            return current_series, StationarityResult(
                is_stationary=True, 
                p_value=p_val, 
                adf_statistic=adf_stat, 
                critical_values=crit_vals, 
                d=d
            )
        
        # Apply differencing if not stationary
        if d < max_d:
            d += 1
            # Ensure the result is treated as a pd.Series
            diff_series = current_series.diff().dropna()
            if isinstance(diff_series, pd.Series):
                current_series = diff_series
            else:
                break
        else:
            break

    return current_series, StationarityResult(
        is_stationary=False, 
        p_value=p_val, 
        adf_statistic=adf_stat, 
        critical_values=crit_vals, 
        d=d
    )
