import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Prévision des Ventes", layout="wide")
st.title("📈 Prévision des Ventes Mensuelles avec ARIMA")

# Step 1: File Upload
uploaded_file = st.file_uploader("📤 Importer le fichier Excel des ventes", type=["xlsx"])

if uploaded_file is not None:
    vente_ligne = pd.read_excel(uploaded_file)
    st.subheader("📋 Aperçu des données originales")
    st.dataframe(vente_ligne.head())

    # Clean the data
    vente_ligne1 = vente_ligne.copy()
    vente_ligne1 = vente_ligne1.drop(columns=vente_ligne1.columns[2:10])
    vente_ligne1 = vente_ligne1.drop(columns=vente_ligne1.columns[3:])
    vente_ligne1 = vente_ligne1.drop(columns=vente_ligne1.columns[0])
    vente_ligne1 = vente_ligne1[vente_ligne1["Montant"] != 0]

    # Convert date column
    vente_ligne1['Date Comptabilisation'] = pd.to_datetime(vente_ligne1['Date Comptabilisation'])
    vente_ligne1['YearMonth'] = vente_ligne1['Date Comptabilisation'].dt.to_period('M').astype(str)

    # Monthly aggregated sales
    monthly_grouped = vente_ligne1.groupby('YearMonth')['Montant'].sum().reset_index()

    st.subheader("📊 Données mensuelles agrégées")
    st.dataframe(monthly_grouped)

    # ADF Test
    adf_result = adfuller(monthly_grouped['Montant'])
    p_value = adf_result[1]
    st.subheader("📉 Test de stationnarité (ADF)")
    st.write(f"**P-value** = {p_value:.5f}")
    if p_value < 0.05:
        st.success("✅ La série est stationnaire.")
    else:
        st.warning("⚠️ La série n’est pas stationnaire.")

    # Plot ACF & PACF
    st.subheader("📌 Analyse des corrélations ACF / PACF")
    col1, col2 = st.columns(2)
    with col1:
        st.write("PACF")
        fig, ax = plt.subplots()
        plot_pacf(monthly_grouped['Montant'], ax=ax)
        st.pyplot(fig)
    with col2:
        st.write("ACF")
        fig, ax = plt.subplots()
        plot_acf(monthly_grouped['Montant'], ax=ax)
        st.pyplot(fig)

    # ARIMA Parameters
    st.subheader("⚙️ Paramètres du modèle ARIMA")
    p = st.number_input("p (PACF)", min_value=0, max_value=10, value=2)
    d = st.number_input("d (Différence)", min_value=0, max_value=2, value=0)
    q = st.number_input("q (ACF)", min_value=0, max_value=10, value=2)

    if st.button("🔮 Lancer la Prévision"):
        # Set index and resample
        data = vente_ligne1.copy()
        data.set_index('Date Comptabilisation', inplace=True)
        monthly_sales = data['Montant'].resample('M').sum()
        mean_sales = monthly_sales[monthly_sales != 0].mean()
        monthly_sales.replace(0, mean_sales, inplace=True)

        # ARIMA Model
        model = ARIMA(monthly_sales, order=(p, d, q))
        model_fit = model.fit()

        # Forecast
        forecast_steps = 12
        forecast = model_fit.get_forecast(steps=forecast_steps)
        forecast_values = forecast.predicted_mean
        last_date = monthly_sales.index[-1]
        forecast_index = pd.date_range(start=last_date, periods=forecast_steps + 1, freq='M')[1:]
        forecast_series = pd.Series(forecast_values, index=forecast_index)

        st.success("✅ Prévision effectuée avec succès !")
        st.subheader("📈 Résultat de la prévision")
        st.dataframe(forecast_series)

        # Plot forecast
        st.subheader("📉 Visualisation des prévisions")
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.plot(monthly_sales, label='Historique des ventes')
        ax.plot(forecast_series, color='red', label='Prévision (12 mois)')
        ax.set_xlabel('Date')
        ax.set_ylabel('Montant des ventes')
        ax.set_title('Prévision des ventes mensuelles avec ARIMA')
        ax.legend()
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)
        plt.grid(True)
        st.pyplot(fig)
else:
    st.info("Veuillez importer un fichier Excel pour commencer.")
