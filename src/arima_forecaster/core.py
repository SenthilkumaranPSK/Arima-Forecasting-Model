import pandas as pd
import numpy as np
from typing import Optional, Tuple, Any, Dict, List, cast
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults

from .preprocessing import prepare_time_series, check_stationarity_detailed, StationarityResult
from .tuning import find_optimal_params
from .analysis import diagnose_residuals, plot_forecast, plot_lag_analysis

class ARIMAForecaster:
    """
    The primary orchestrator for the ARIMA forecasting pipeline.
    Integrates preprocessing, parameter tuning, model fitting, and diagnostics.
    """
    
    def __init__(self):
        """Initializes the forecaster with empty state."""
        self.series: Optional[pd.Series] = None
        self.model: Optional[ARIMA] = None
        self.results: Optional[ARIMAResults] = None
        self.order: Optional[Tuple[int, int, int]] = None
        self.metadata: Dict[str, Any] = {}

    def prepare_data(
        self, 
        df: pd.DataFrame, 
        date_col: str, 
        target_col: str, 
        freq: Optional[str] = None
    ) -> pd.Series:
        """
        Cleans and prepares the input DataFrame.
        
        Args:
            df: Input DataFrame.
            date_col: Name of the date column.
            target_col: Name of the target value column.
            freq: Optional pandas frequency string.
            
        Returns:
            pd.Series: Cleaned time series.
        """
        self.series = prepare_time_series(df, date_col, target_col, freq=freq)
        self.metadata['original_length'] = len(self.series)
        return self.series

    def auto_fit(
        self, 
        p_range: List[int] = [0, 1, 2, 3], 
        q_range: List[int] = [0, 1, 2, 3],
        max_d: int = 2
    ) -> ARIMAResults:
        """
        Automatically runs the full pipeline:
        1. Prepares/validates stationarity to find 'd'.
        2. Grid searches for optimal 'p' and 'q' using AIC.
        3. Fits the final model.

        Args:
            p_range: List of AR orders to test.
            q_range: List of MA orders to test.
            max_d: Maximum differencing steps allowed.

        Returns:
            ARIMAResults: The fitted statsmodels results object.
        """
        if self.series is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")
        
        series = self.series # Type narrowing

        # 1. Stationarity Check (Determine d)
        print("--- Step 1: Stationarity Analysis ---")
        _, stat_result = check_stationarity_detailed(series, max_d=max_d)
        d = stat_result.d
        self.metadata['stationarity'] = stat_result
        print(f"Determined differencing parameter d = {d}")

        # 2. Parameter Tuning (Determine p and q)
        print("\n--- Step 2: Parameter Tuning (AIC Grid Search) ---")
        best_order, tuning_summary = find_optimal_params(
            series, p_range=p_range, d_range=[d], q_range=q_range
        )
        self.order = best_order
        self.metadata['tuning_summary'] = tuning_summary
        
        # 3. Final Fit
        return self.fit(best_order)

    def fit(self, order: Tuple[int, int, int]) -> ARIMAResults:
        """
        Fits an ARIMA model with a specific order.

        Args:
            order: (p, d, q) tuple.

        Returns:
            ARIMAResults: The fitted results object.
        """
        if self.series is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")

        series = self.series # Type narrowing
        
        print(f"\n--- Step 3: Fitting ARIMA{order} ---")
        self.order = order
        self.model = ARIMA(series, order=order)
        fitted_results = self.model.fit()
        # Cast to ARIMAResults to satisfy Pylance
        self.results = cast(ARIMAResults, fitted_results)
        
        print("Model fitted successfully.")
        return self.results

    def forecast(self, steps: int = 10, alpha: float = 0.05) -> pd.DataFrame:
        """
        Generates forecasts for future time steps.

        Args:
            steps: Number of steps to forecast.
            alpha: Significance level for confidence intervals.

        Returns:
            pd.DataFrame: Forecast values and confidence intervals.
        """
        if self.results is None:
            raise RuntimeError("Model not fitted. Call fit() or auto_fit() first.")

        forecast_res = self.results.get_forecast(steps=steps)
        forecast_df = forecast_res.summary_frame(alpha=alpha)
        
        # Rename columns for clarity
        forecast_df.columns = ['mean', 'mean_se', 'lower_ci', 'upper_ci']
        
        return forecast_df

    def run_diagnostics(self):
        """
        Performs residual analysis and lag visualizations for the fitted model.
        """
        # Capture into local variables for reliable type narrowing
        results = self.results
        series = self.series
        order = self.order

        if results is None or series is None or order is None:
            raise RuntimeError("Model not fitted correctly. Ensure fit() or auto_fit() succeeded.")

        print("\n--- Running Model Diagnostics ---")
        diagnose_residuals(results)
        
        print("\n--- Visualizing ACF/PACF of Stationary Data ---")
        # order[1] is 'd'
        stationary_series, _ = check_stationarity_detailed(series, max_d=order[1])
        plot_lag_analysis(stationary_series)

    def plot_results(self, forecast_steps: int = 10):
        """
        Plots the historical data alongside the future forecast.
        """
        series = self.series
        if series is None:
             raise RuntimeError("Data not prepared. Call prepare_data() first.")

        forecast_df = self.forecast(steps=forecast_steps)
        plot_forecast(
            actual=series, 
            forecast=forecast_df['mean'], 
            conf_int=forecast_df[['lower_ci', 'upper_ci']]
        )
