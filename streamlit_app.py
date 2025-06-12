import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.arima.model import ARIMA
from datetime import timedelta

st.title("📈 Prévision des Ventes avec ARIMA")
st.write("Chargez un fichier contenant une colonne de date et une colonne de valeurs pour effectuer une prévision ARIMA.")

uploaded_file = st.file_uploader("📂 Charger un fichier (CSV, Excel, ou TXT)", type=["csv", "xls", "xlsx", "txt"])

if uploaded_file:
    # Lire le fichier
    file_name = uploaded_file.name.lower()
    if file_name.endswith(".csv") or file_name.endswith(".txt"):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file)

    st.subheader("📋 Aperçu des Données")
    st.write(data.head())

    # Choisir les colonnes
    st.subheader("🛠️ Sélection des Colonnes")
    columns = list(data.columns)
    date_col = st.selectbox("Colonne de Date", columns)
    value_col = st.selectbox("Colonne de Valeurs (Montant, Ventes, etc.)", columns)

    try:
        data[date_col] = pd.to_datetime(data[date_col])
        data = data[[date_col, value_col]].dropna()
        data = data.rename(columns={date_col: "Date", value_col: "Valeur"})
        data = data.set_index("Date")
        monthly_data = data.resample("M").sum()

        # Remplacer les zéros par la moyenne (optionnel mais utile)
        mean_val = monthly_data[monthly_data["Valeur"] != 0]["Valeur"].mean()
        monthly_data["Valeur"].replace(0, mean_val, inplace=True)

        st.subheader("📊 Série Temporelle (Agrégée Mensuellement)")
        st.line_chart(monthly_data["Valeur"])

        # ARIMA params
        st.subheader("⚙️ Paramètres du modèle ARIMA")
        p = st.number_input("p (PACF)", min_value=0, max_value=10, value=2)
        d = st.number_input("d (Différence)", min_value=0, max_value=2, value=0)
        q = st.number_input("q (ACF)", min_value=0, max_value=10, value=2)
        steps = st.number_input("Périodes à prédire", min_value=1, max_value=36, value=12)

        if st.button("🔮 Lancer la Prévision"):
            model = ARIMA(monthly_data["Valeur"], order=(p, d, q))
            model_fit = model.fit()

            forecast = model_fit.get_forecast(steps=steps)
            forecast_values = forecast.predicted_mean
            last_date = monthly_data.index[-1]
            forecast_index = pd.date_range(start=last_date + timedelta(days=1), periods=steps, freq='M')
            forecast_series = pd.Series(forecast_values, index=forecast_index)

            st.success("✅ Prévision effectuée avec succès !")
            st.subheader("📈 Résultat de la prévision")
            st.dataframe(forecast_series)

            # Visualisation
            st.subheader("📉 Visualisation des prévisions")
            fig, ax = plt.subplots(figsize=(14, 5))
            ax.plot(monthly_data["Valeur"], label="Historique")
            ax.plot(forecast_series, color="red", label="Prévision")
            ax.set_title("Prévision ARIMA des ventes mensuelles")
            ax.set_xlabel("Date")
            ax.set_ylabel("Valeurs")
            ax.legend()
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.xticks(rotation=45)
            plt.grid(True)
            st.pyplot(fig)

            # Téléchargement
            forecast_csv = forecast_series.reset_index().rename(columns={"index": "Date", 0: "Prévision"}).to_csv(index=False)
            st.download_button("📥 Télécharger les prévisions", forecast_csv, file_name="prevision_arima.csv", mime="text/csv")

    except Exception as e:
        st.error(f"❌ Erreur : {str(e)}")
else:
    st.info("Veuillez charger un fichier pour commencer.")
