# download_data.py
import pandas as pd
import numpy as np

def generate_dpe_dataset(n_samples=5000, output_file="dataset_dpe.csv"):
    print(f"⚙️ Génération d'un dataset DPE réaliste ({n_samples} logements)...")
    
    np.random.seed(42)
    
    # 1. Caractéristiques des logements
    type_batiment = np.random.choice(['Maison', 'Appartement'], size=n_samples, p=[0.45, 0.55])
    surface_habitable = np.random.normal(loc=75, scale=30, size=n_samples).clip(15, 250).round(1)
    annee_construction = np.random.randint(1950, 2024, size=n_samples)
    
    type_chauffage = np.random.choice(
        ['Électricité', 'Gaz', 'Fioul', 'Pompe à chaleur', 'Bois'], 
        size=n_samples, 
        p=[0.40, 0.30, 0.10, 0.15, 0.05]
    )
    
    qualite_isolation = np.random.choice(
        ['Bonne', 'Moyenne', 'Insuffisante'], 
        size=n_samples, 
        p=[0.30, 0.45, 0.25]
    )
    
    zone_climatique = np.random.choice(['H1', 'H2', 'H3'], size=n_samples, p=[0.50, 0.35, 0.15])
    
    # 2. Formule physique réaliste pour calculer la consommation d'énergie (Target)
    # Plus le bâtiment est ancien ou mal isolé, plus la consommation augmente
    base_conso = 180.0
    
    # Impact de l'année de construction (avant 1975 = passoire thermique)
    age_factor = np.where(annee_construction < 1975, 70.0, 
                 np.where(annee_construction < 2000, 30.0, -40.0))
    
    # Impact du mode de chauffage
    chauffage_impact = {
        'Électricité': 20.0,
        'Fioul': 50.0,
        'Gaz': 10.0,
        'Bois': -10.0,
        'Pompe à chaleur': -60.0
    }
    chauffage_factor = np.array([chauffage_impact[c] for c in type_chauffage])
    
    # Impact de l'isolation
    isolation_impact = {'Bonne': -50.0, 'Moyenne': 0.0, 'Insuffisante': 80.0}
    isolation_factor = np.array([isolation_impact[i] for i in qualite_isolation])
    
    # Bruit aléatoire (variations réelles d'usage)
    noise = np.random.normal(0, 15, size=n_samples)
    
    # Calcul final de la consommation en kWh/m²/an
    conso_energie = base_conso + age_factor + chauffage_factor + isolation_factor + noise
    conso_energie = conso_energie.clip(30, 500).round(1) # Valeurs réalistes DPE
    
    # 3. Assemblage du DataFrame
    df = pd.DataFrame({
        'surface_habitable': surface_habitable,
        'annee_construction': annee_construction,
        'type_batiment': type_batiment,
        'type_chauffage': type_chauffage,
        'qualite_isolation': qualite_isolation,
        'zone_climatique': zone_climatique,
        'conso_energie_kwh_m2_an': conso_energie  # Target à prédire
    })
    
    df.to_csv(output_file, index=False)
    print(f"✅ Fichier '{output_file}' généré avec succès ! ({len(df)} lignes)")
    print("\nAperçu des 5 premières lignes :")
    print(df.head())

if __name__ == "__main__":
    generate_dpe_dataset(n_samples=5000)