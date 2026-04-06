import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# =========================
# 1. Paramètres du projet
# =========================
DATA_PATH = "football.csv"
TARGET = "target"

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

MODEL_PATH = "model_logistic.pkl"
SCALER_PATH = "scaler.pkl"
IMPUTER_PATH = "imputer.pkl"

# =========================
# 2. Charger les données
# =========================
df = pd.read_csv(DATA_PATH)

print("Dimensions du dataset :", df.shape)
print("Colonnes disponibles :", df.columns.tolist())

# Vérifier que les colonnes attendues existent
missing_cols = [col for col in FEATURES + [TARGET] if col not in df.columns]
if missing_cols:
    raise ValueError(f"Colonnes manquantes dans le dataset : {missing_cols}")

# =========================
# 3. Séparer X et y
# =========================
X = df[FEATURES].copy()
y = df[TARGET].copy()

# =========================
# 4. Split train / test
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Taille X_train :", X_train.shape)
print("Taille X_test  :", X_test.shape)

# =========================
# 5. Imputation médiane
# =========================
imputer = SimpleImputer(strategy="median")
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# =========================
# 6. Standardisation
# =========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# =========================
# 7. Entraînement du modèle
# =========================
model = LogisticRegression(
    C=0.01,
    solver="liblinear",
    max_iter=1000
)

model.fit(X_train_scaled, y_train)

# =========================
# 8. Évaluation rapide
# =========================
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy test : {accuracy:.4f}")
print("\nClassification report :")
print(classification_report(y_test, y_pred))

# =========================
# 9. Sauvegarde des artefacts
# =========================
joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
joblib.dump(imputer, IMPUTER_PATH)

print("\nArtefacts sauvegardés :")
print(f"- {MODEL_PATH}")
print(f"- {SCALER_PATH}")
print(f"- {IMPUTER_PATH}")