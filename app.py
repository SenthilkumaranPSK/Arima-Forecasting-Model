import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Ensure the local arima_forecaster package is importable
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from arima_forecaster import ARIMAForecaster

# --- App Configuration ---
st.set_page_config(page_title="ARIMA Forecast Pro", layout="wide")

st.title("📈 ARIMA Forecasting Dashboard")
st.markdown("""
Upload your time series data (CSV), select your columns, and let the ARIMA module automatically 
find the best parameters and generate a forecast.
""")

# --- Sidebar: Configuration ---
st.sidebar.header("1. Upload & Configure")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("File uploaded successfully!")
    
    # Column Selection
    columns = df.columns.tolist()
    date_col = st.sidebar.selectbox("Select Date Column", options=columns)
    target_col = st.sidebar.selectbox("Select Target Column", options=columns)
    
    # Model Parameters
    st.sidebar.header("2. Tuning Parameters")
    p_max = st.sidebar.slider("Max AR (p)", 0, 5, 3)
    q_max = st.sidebar.slider("Max MA (q)", 0, 5, 3)
    forecast_steps = st.sidebar.number_input("Forecast Steps", min_value=1, value=14)
    
    # --- Main Area: Data Preview ---
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Data Preview")
        st.write(df.head())
    
    with col2:
        st.subheader("Raw Data Plot")
        # Quick plot of the raw target
        try:
            temp_df = df.copy()
            # Handle potential spaces AND colons which break Altair/Streamlit charts
            # Replace ':' with '_' because Altair uses ':' for type encoding
            temp_df.columns = [c.strip().replace(":", "_") for c in temp_df.columns]
            
            current_date_col = date_col.strip().replace(":", "_")
            current_target_col = target_col.strip().replace(":", "_")

            if current_date_col in temp_df.columns and current_target_col in temp_df.columns:
                temp_df[current_date_col] = pd.to_datetime(temp_df[current_date_col], errors='coerce')
                temp_df = temp_df.dropna(subset=[current_date_col])
                temp_df = temp_df.set_index(current_date_col)
                # Ensure target is numeric for plotting
                temp_df[current_target_col] = pd.to_numeric(temp_df[current_target_col], errors='coerce')
                temp_df = temp_df.dropna(subset=[current_target_col])
                st.line_chart(temp_df[current_target_col])
            else:
                st.error(f"Selected columns not found in cleaned column names: {temp_df.columns.tolist()}")
        except Exception as e:
            st.error(f"Error plotting data: {e}")

    # --- Run Forecast ---
    if st.sidebar.button("🚀 Run Forecast"):
        with st.spinner("Analyzing stationarity and tuning ARIMA parameters..."):
            try:
                # Initialize Forecaster
                forecaster = ARIMAForecaster()
                
                # Prepare Data
                forecaster.prepare_data(df, date_col=date_col, target_col=target_col)
                
                # Auto Fit
                p_range = list(range(p_max + 1))
                q_range = list(range(q_max + 1))
                forecaster.auto_fit(p_range=p_range, q_range=q_range)
                
                # --- Results Display ---
                st.divider()
                st.header("📊 Forecasting Results")
                
                res_col1, res_col2 = st.columns([1, 2])
                
                with res_col1:
                    st.subheader("Model Summary")
                    st.write(f"**Best Order (p, d, q):** {forecaster.order}")
                    st.write(f"**AIC:** {forecaster.results.aic:.2f}")
                    st.write(f"**BIC:** {forecaster.results.bic:.2f}")
                    
                    st.subheader("Forecast Data")
                    forecast_df = forecaster.forecast(steps=forecast_steps)
                    st.write(forecast_df)

                with res_col2:
                    st.subheader("Actual vs Forecast")
                    # We'll use matplotlib directly for the themed plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(forecaster.series, label="Historical", color="navy")
                    ax.plot(forecast_df['mean'], label="Forecast", color="crimson", linestyle="--")
                    ax.fill_between(
                        forecast_df.index, 
                        forecast_df['lower_ci'], 
                        forecast_df['upper_ci'], 
                        color='crimson', alpha=0.1, label="95% CI"
                    )
                    ax.set_title("Historical Observations vs. Predicted Future")
                    ax.legend()
                    st.pyplot(fig)

                # --- Diagnostics ---
                st.divider()
                st.subheader("🔍 Residual Diagnostics")
                diag_fig = forecaster.results.plot_diagnostics(figsize=(12, 8))
                st.pyplot(diag_fig)
                
            except Exception as e:
                st.error(f"An error occurred during forecasting: {e}")
                st.info("Check if your date column is formatted correctly and has a consistent frequency.")

else:
    st.info("Please upload a CSV file in the sidebar to get started.")
    # Show a small example
    st.write("### Example CSV Format")
    example_df = pd.DataFrame({
        "Date": ["2023-01-01", "2023-01-02", "2023-01-03"],
        "Sales": [10.5, 12.2, 11.8]
    })
    st.table(example_df)
    st.caption("Ensure your CSV has no empty rows and follows a consistent date format.")
