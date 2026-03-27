import pandas as pd
import numpy as np
import warnings
from typing import Tuple, List, Dict, Any, Optional
from statsmodels.tsa.arima.model import ARIMA

def find_optimal_params(
    series: pd.Series, 
    p_range: List[int] = [0, 1, 2, 3], 
    d_range: List[int] = [0, 1, 2], 
    q_range: List[int] = [0, 1, 2, 3],
    enforce_stationarity: bool = True,
    enforce_invertibility: bool = True
) -> Tuple[Tuple[int, int, int], pd.DataFrame]:
    """
    Performs a grid search to find the optimal ARIMA (p, d, q) parameters 
    based on the lowest AIC (Akaike Information Criterion) value.

    Args:
        series: The input time series data.
        p_range: List of integers to test for the AR (p) component.
        d_range: List of integers to test for the Integrated (d) component.
        q_range: List of integers to test for the MA (q) component.
        enforce_stationarity: Whether to enforce stationarity on the AR coefficients.
        enforce_invertibility: Whether to enforce invertibility on the MA coefficients.

    Returns:
        Tuple[Tuple[int, int, int], pd.DataFrame]: 
            1. The best (p, d, q) order found.
            2. A summary DataFrame of all tested combinations and their AIC scores.
    """
    results_list = []
    best_aic = float("inf")
    best_order = (0, 0, 0)

    # Suppress convergence and estimation warnings from statsmodels during tuning
    warnings.filterwarnings("ignore")

    print(f"Starting ARIMA grid search (p: {p_range}, d: {d_range}, q: {q_range})...")

    for p in p_range:
        for d in d_range:
            for q in q_range:
                order = (p, d, q)
                try:
                    # Fit ARIMA model
                    model = ARIMA(
                        series, 
                        order=order, 
                        enforce_stationarity=enforce_stationarity, 
                        enforce_invertibility=enforce_invertibility
                    )
                    model_fit = model.fit()
                    
                    aic = model_fit.aic
                    results_list.append({"order": order, "aic": aic})

                    if aic < best_aic:
                        best_aic = aic
                        best_order = order
                        
                except Exception as e:
                    # Handle cases where the model fails to converge or is mathematically invalid
                    results_list.append({"order": order, "aic": np.nan})
                    continue

    # Restore warnings
    warnings.filterwarnings("default")

    # Create tuning summary
    summary_df = pd.DataFrame(results_list).sort_values(by="aic").reset_index(drop=True)
    
    if summary_df.dropna(subset=['aic']).empty:
        raise RuntimeError("Grid search failed to find any valid ARIMA models. Check your data quality.")

    print(f"Optimal order found: {best_order} with AIC: {best_aic:.2f}")
    
    return best_order, summary_df

def find_optimal_pq(
    series: pd.Series, 
    d: int,
    max_p: int = 5, 
    max_q: int = 5
) -> Tuple[int, int]:
    """
    A helper function to find optimal p and q for a fixed differencing parameter d.
    This is useful when the stationarity (d) is pre-determined by an ADF test.

    Args:
        series: The time series data.
        d: The fixed differencing order.
        max_p: Maximum AR order to test.
        max_q: Maximum MA order to test.

    Returns:
        Tuple[int, int]: Best (p, q) combination.
    """
    p_range = list(range(max_p + 1))
    q_range = list(range(max_q + 1))
    
    best_order, _ = find_optimal_params(series, p_range=p_range, d_range=[d], q_range=q_range)
    
    return best_order[0], best_order[2]
