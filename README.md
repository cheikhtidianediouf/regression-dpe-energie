Markdown
# ⚡ DPE Energy Consumption Prediction - MLOps Pipeline

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Regressor-orange.svg)](https://xgboost.readthedocs.io/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-blueviolet.svg)](https://www.docker.com/)

## 🎯 Aperçu du Projet
Ce projet implémente un modèle de **Machine Learning de régression** pour prédire la consommation d'énergie primaire d'un logement ($kWh/m^2/an$) à partir des données officielles du **DPE (Diagnostic de Performance Énergétique)** de l'ADEME.

Au-delà de la modélisation prédictive, ce projet intègre les meilleures pratiques **MLOps** :
* Pipeline de pré-traitement robuste (Scikit-Learn `Pipeline`, `ColumnTransformer`, `StandardScaler`, `OneHotEncoder`).
* Modèle d'ensemble haute performance (`XGBoost Regressor`).
* API REST industrialisée avec **FastAPI** et validation de données (`Pydantic`).
* Conteneurisation prête pour le déploiement (`Docker`).

---

## 📊 Performances du Modèle

Le modèle a été évalué sur un jeu de test indépendant (20% des données) :

| Métrique | Score | Interprétation |
| :--- | :--- | :--- |
| **$R^2$ Score** | **0.9594** | Le modèle explique près de 96% de la variance de la consommation. |
| **MAE** (Erreur Absolue Moyenne) | **11.93 kWh/m²/an** | Erreur moyenne très faible sur les prédictions. |
| **RMSE** | **14.98 kWh/m²/an** | Robustesse face aux valeurs atypiques. |

---

## 🛠️ Stack Technique
* **Langage :** Python 3.12
* **Data Science & ML :** Pandas, NumPy, Scikit-Learn, XGBoost, Joblib
* **API Backend :** FastAPI, Uvicorn, Pydantic
* **DevOps / MLOps :** Git, Docker

---

## 📁 Structure du Projet

```text
regression-dpe-energie/
│
├── dataset_dpe.csv              # Données tabulaires (générées / nettoyées)
├── download_data.py             # Script de génération / récupération des données
├── train_model.py               # Script d'entraînement, preprocessing et métriques
├── model_dpe_xgboost.pkl        # Modèle sérialisé (Pipeline complet)
├── main.py                      # Application FastAPI (Endpoints d'inférence)
├── Dockerfile                   # Configuration pour la conteneurisation
├── requirements.txt             # Dépendances du projet
└── README.md                    # Documentation du projet
🚀 Installation et Utilisation Locale
1. Cloner le dépôt
Bash
git clone [https://github.com/cheikhtidianediouf/regression-dpe-energie.git](https://github.com/cheikhtidianediouf/regression-dpe-energie.git)
cd regression-dpe-energie
2. Installer les dépendances
Bash
pip install -r requirements.txt
3. Entraîner le modèle (Optionnel si le .pkl est déjà présent)
Bash
python train_model.py
4. Lancer l'API FastAPI
Bash
uvicorn main:app --reload
L'API est alors accessible sur http://127.0.0.1:8000. Vous pouvez tester l'interface interactive Swagger UI sur http://127.0.0.1:8000/docs.

📡 Exemple de Requête API (/predict)
Requête JSON :

JSON
{
  "surface_habitable": 75.0,
  "annee_construction": 2005,
  "type_batiment": "Appartement",
  "type_chauffage": "Gaz",
  "qualite_isolation": "Moyenne",
  "zone_climatique": "H1"
}
Réponse JSON :

JSON
{
  "logement": {
    "surface_habitable": 75.0,
    "annee_construction": 2005,
    "type_batiment": "Appartement",
    "type_chauffage": "Gaz",
    "qualite_isolation": "Moyenne",
    "zone_climatique": "H1"
  },
  "consommation_estimee_kwh_m2_an": 152.1
}
🐳 Conteneurisation Docker
Pour construire et exécuter le projet dans un conteneur Docker :

Bash
# Construire l'image Docker
docker build -t dpe-regression-api .

# Lancer le conteneur
docker run -p 8000:8000 dpe-regression-api