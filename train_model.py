# train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib

def train_dpe_model():
    print("🚀 Chargement du dataset DPE...")
    df = pd.read_csv("dataset_dpe.csv")
    
    # 1. Séparation Features (X) / Target (y)
    X = df.drop(columns=['conso_energie_kwh_m2_an'])
    y = df['conso_energie_kwh_m2_an']
    
    # 2. Train / Test Split (80% entraînement, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"📊 Données d'entraînement : {len(X_train)} lignes | Données de test : {len(X_test)} lignes")
    
    # 3. Identification des colonnes numériques et catégorielles
    numeric_features = ['surface_habitable', 'annee_construction']
    categorical_features = ['type_batiment', 'type_chauffage', 'qualite_isolation', 'zone_climatique']
    
    # 4. Pipeline de pré-traitement
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )
    
    # 5. Modèle XGBoost Regressor
    model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    
    # Pipeline complet (Preprocessing + Modèle)
    pipeline = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('regressor', model)
        ]
    )
    
    # 6. Entraînement
    print("⏳ Entraînement du modèle XGBoost en cours...")
    pipeline.fit(X_train, y_train)
    print("✅ Entraînement terminé avec succès !")
    
    # 7. Évaluation sur le jeu de test
    y_pred = pipeline.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print("\n--- 📈 Métriques de Performance du Modèle ---")
    print(f"• MAE (Erreur Absolue Moyenne) : {mae:.2f} kWh/m²/an")
    print(f"• RMSE (Racine de l'erreur quadratique) : {rmse:.2f} kWh/m²/an")
    print(f"• R² Score (Coefficient de détermination) : {r2:.4f} (Proche de 1 = excellent)")
    
    # 8. Sauvegarde du pipeline entraîné
    model_filename = "model_dpe_xgboost.pkl"
    joblib.dump(pipeline, model_filename)
    print(f"\n💾 Modèle sauvegardé avec succès sous : {model_filename}")

if __name__ == "__main__":
    train_dpe_model()