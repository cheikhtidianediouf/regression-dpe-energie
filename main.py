# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(
    title="API de Prédiction de Consommation Énergétique (DPE)",
    description="API MLOps de régression pour estimer la consommation d'un logement (kWh/m²/an) via XGBoost",
    version="1.0"
)

# Chargement du modèle entraîné (Pipeline complet Preprocessor + XGBoost)
model = joblib.load("model_dpe_xgboost.pkl")

# Schéma de validation Pydantic pour s'assurer de la qualité des données entrantes
class HouseFeatures(BaseModel):
    surface_habitable: float
    annee_construction: int
    type_batiment: str        # Ex: "Maison" ou "Appartement"
    type_chauffage: str       # Ex: "Électricité", "Gaz", "Fioul", "Pompe à chaleur", "Bois"
    qualite_isolation: str    # Ex: "Bonne", "Moyenne", "Insuffisante"
    zone_climatique: str      # Ex: "H1", "H2", "H3"

@app.get("/")
def home():
    return {
        "message": "Bienvenue sur l'API MLOps de prédiction DPE !",
        "documentation": "/docs"
    }

@app.post("/predict")
def predict_energy(features: HouseFeatures):
    # Transformation de la requête JSON en DataFrame Pandas pour le pipeline
    input_data = pd.DataFrame([features.dict()])
    
    # Inférence avec le modèle XGBoost
    prediction = model.predict(input_data)[0]
    
    return {
        "logement": features.dict(),
        "consommation_estimee_kwh_m2_an": round(float(prediction), 2)
    }