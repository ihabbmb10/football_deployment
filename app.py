from flask import Flask, request, jsonify
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

# =========================
# 1. Charger les artefacts
# =========================
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

# =========================
# 2. Route de test
# =========================
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "API Football opérationnelle",
        "endpoint": "/predict"
    })

# =========================
# 3. Route de prédiction
# =========================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if data is None:
            return jsonify({
                "error": "Aucun JSON reçu",
                "code": 400
            }), 400

        # Vérifier les colonnes manquantes
        missing_features = [f for f in FEATURES if f not in data]
        if missing_features:
            return jsonify({
                "error": "Features manquantes",
                "missing_features": missing_features,
                "code": 400
            }), 400

        # Construire DataFrame dans le bon ordre
        input_df = pd.DataFrame([[data[f] for f in FEATURES]], columns=FEATURES)

        # Vérification des types numériques
        for col in FEATURES:
            value = input_df.at[0, col]
            if value is None:
                input_df.at[0, col] = np.nan
            elif not isinstance(value, (int, float)):
                return jsonify({
                    "error": f"Type invalide pour {col}: attendu int/float, reçu {type(value).__name__}",
                    "code": 422
                }), 422

        # Prétraitement
        input_imputed = imputer.transform(input_df)
        input_scaled = scaler.transform(input_imputed)

        # Prédiction
        prediction = int(model.predict(input_scaled)[0])
        probability = float(model.predict_proba(input_scaled)[0][1])

        label = "Victoire domicile" if prediction == 1 else "Défaite ou nul domicile"

        response = {
            "prediction": prediction,
            "probability": round(probability, 4),
            "label": label
        }

        # Ajouter un avertissement si proba trop proche de 0.5
        if 0.45 <= probability <= 0.55:
            response["warning"] = "Prédiction peu fiable : probabilité proche de 0.5"

        return jsonify(response), 200

    except Exception as e:
        return jsonify({
            "error": str(e),
            "code": 500
        }), 500

# =========================
# 4. Lancement
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)