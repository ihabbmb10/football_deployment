import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Charger les artefacts
model = joblib.load("model_logistic.pkl")
scaler = joblib.load("scaler.pkl")
imputer = joblib.load("imputer.pkl")

FEATURES = [
    "rank_change_home",
    "rank_change_away",
    "home_goals_mean",
    "home_goals_mean_l5",
    "home_goals_suf_mean",
    "home_goals_suf_mean_l5",
    "home_rank_mean",
    "home_rank_mean_l5",
    "home_points_mean",
    "home_points_mean_l5",
    "away_goals_mean",
    "away_goals_mean_l5",
    "away_goals_suf_mean",
    "away_goals_suf_mean_l5",
    "away_rank_mean",
    "away_rank_mean_l5",
    "away_points_mean",
    "away_points_mean_l5"
]

st.title("Prédiction des scores de matchs de football")
st.write("Entrer les statistiques du match pour obtenir une prédiction.")

inputs = {}

col1, col2 = st.columns(2)

for i, feature in enumerate(FEATURES):
    if i % 2 == 0:
        with col1:
            inputs[feature] = st.number_input(feature, value=0.0, format="%.4f")
    else:
        with col2:
            inputs[feature] = st.number_input(feature, value=0.0, format="%.4f")

if st.button("Prédire"):
    input_df = pd.DataFrame([inputs])

    input_imputed = imputer.transform(input_df)
    input_scaled = scaler.transform(input_imputed)

    prediction = int(model.predict(input_scaled)[0])
    probability = float(model.predict_proba(input_scaled)[0][1])

    label = "Victoire domicile" if prediction == 1 else "Défaite ou nul domicile"

    st.subheader("Résultat")
    st.write("Prédiction :", label)
    st.write("Probabilité :", round(probability, 4))

    if 0.45 <= probability <= 0.55:
        st.warning("Prédiction peu fiable : la probabilité est proche de 0.5.")