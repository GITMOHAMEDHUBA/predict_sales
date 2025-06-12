import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from datetime import timedelta

# Title
st.title("📈 Forecasting with ARIMA")
st.write("Upload your time series data and forecast future values using the ARIMA model.")

# Upload file (supports CSV, Excel, and TXT)
uploaded_file = st.file_uploader("Upload a file", type=["csv", "xls", "xlsx", "txt"])

if uploaded_file:
    file_name = uploaded_file.name.lower()
    
    if file_name.endswith(".csv") or file_name.endswith(".txt"):
        data = pd.read_csv(uploaded_file)
    elif file_name.endswith((".xls", ".xlsx")):
        data = pd.read_excel(uploaded_file)
    else:
        st.error("Unsupported file type.")
        st.stop()

    st.subheader("📋 Uploaded Data Preview")
    st.write(data.head())

    # Let user choose the date and sales columns
    st.subheader("🛠️ Select Columns")
    columns = list(data.columns)
    date_col = st.selectbox("Select the column representing **Date**", columns)
    value_col = st.selectbox("Select the column representing **Sales/Values**", columns)

    try:
        # Convert and prepare data
        data[date_col] = pd.to_datetime(data[date_col])
        data = data[[date_col, value_col]].sort_values(by=date_col)
        data.set_index(date_col, inplace=True)

        # Plot original data
        st.subheader("📊 Historical Data")
        st.line_chart(data[value_col])

        # ARIMA config
        st.subheader("⚙️ ARIMA Configuration")
        p = st.number_input("AR term (p)", min_value=0, max_value=5, value=1)
        d = st.number_input("Differencing order (d)", min_value=0, max_value=2, value=1)
        q = st.number_input("MA term (q)", min_value=0, max_value=5, value=1)
        steps = st.number_input("Forecast steps", min_value=1, max_value=100, value=12)

        if st.button("🚀 Run Forecast"):
            with st.spinner("Training the ARIMA model..."):
                model = ARIMA(data[value_col], order=(p, d, q))
                fitted_model = model.fit()

            forecast = fitted_model.forecast(steps=steps)
            last_date = data.index[-1]
            freq = pd.infer_freq(data.index) or 'D'
            future_dates = pd.date_range(start=last_date + pd.Timedelta(1, unit='D'), periods=steps, freq=freq)
            forecast_df = pd.DataFrame({'Forecast': forecast}, index=future_dates)

            # Combine original and forecast for display
            combined = pd.concat([data[value_col], forecast_df['Forecast']])

            st.subheader("🔮 Forecast Results")
            st.line_chart(combined)

            # Download CSV
            csv = forecast_df.reset_index().rename(columns={'index': 'Date'}).to_csv(index=False)
            st.download_button("📥 Download Forecast CSV", data=csv, file_name='forecast.csv', mime='text/csv')

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
