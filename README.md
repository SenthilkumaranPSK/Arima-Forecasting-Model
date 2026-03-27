# ARIMA Forecasting Module 📈

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)

A robust, enterprise-ready Python module and interactive dashboard for high-precision time series forecasting using the ARIMA (AutoRegressive Integrated Moving Average) model.

## ✨ Why this Module?

Traditional time series analysis requires deep statistical knowledge. This module **democratizes forecasting** by automating the complex math:
*   **Automatic Stationarity Detection**: No more manual ADF testing.
*   **Optimal Hyperparameter Tuning**: Uses AIC/BIC grid search to find $(p, d, q)$ automatically.
*   **Instant Visualization**: Beautiful charts and diagnostic plots included.
*   **Clean API**: Designed for developers to integrate into larger pipelines.

---

## 🚀 Quick Start

### 1. Interactive Dashboard (Recommended)
Launch the Streamlit UI to upload your own CSV files and generate forecasts instantly:
```bash
streamlit run app.py
```

### 2. Python API
Integrate the `ARIMAForecaster` directly into your scripts:

```python
import pandas as pd
from arima_forecaster import ARIMAForecaster

# Initialize and Prepare
forecaster = ARIMAForecaster()
df = pd.read_csv("data.csv")
forecaster.prepare_data(df, date_col='Date', target_col='Sales')

# Auto-fit and Forecast
forecaster.auto_fit()
forecast = forecaster.forecast(steps=14)
print(forecast.head())
```

---

## 🛠️ Installation & Setup

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/SenthilkumaranPSK/Arima-Forecasting-Model.git
    cd Arima-Forecasting-Model
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Local Package Installation**:
    ```bash
    pip install -e .
    ```

---

## 📂 Project Architecture

```text
arima_forecasting_module/
├── app.py              # Streamlit Interactive Dashboard
├── src/                # Core forecasting engine
│   └── arima_forecaster/
│       ├── preprocessing.py  # Cleaning & stationarity
│       ├── core.py           # Main ARIMAForecaster class
│       └── analysis.py       # Plots & diagnostics
├── examples/           # Jupyter notebooks & scripts
├── tests/              # Pytest suite
└── requirements.txt    # Project dependencies
```

---

## 🧪 Development & Testing

We maintain high code quality with automated tests. Run the suite using:
```bash
pytest tests/
```

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

---

**Developed with ❤️ by [Senthilkumaran](https://github.com/SenthilkumaranPSK)**
