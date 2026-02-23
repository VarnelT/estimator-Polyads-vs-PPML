#pip install scikit-learn
import numpy as np
import pandas as pd
import time
import sys
import os
from sklearn.preprocessing import LabelEncoder

# --- CONFIGURATION DU PATH ---
# On ajoute la racine du projet et le dossier src pour garantir l'importation
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

for path in [current_dir, project_root]:
    if path not in sys.path:
        sys.path.append(path)

# --- IMPORTATION DU MODÈLE ---
try:
    # Tentative d'importation directe ou via le dossier src
    try:
        from polyads.model import PolyadEstimator
    except ImportError:
        from src.polyads.model import PolyadEstimator
except ImportError as e:
    print(f"❌ ERREUR CRITIQUE : Impossible de charger PolyadEstimator.\n{e}")
    sys.exit(1)

# --- CONFIGURATION ---
DATA_PATH = "/home/onyxia/work/estimator-Polyads-vs-PPML/data/processed/panel_total_trade.parquet"

# Benchmark obtenu via le script R (02_estimate_ppml.R)
# À mettre à jour si tes résultats R changent légèrement
BENCHMARK_PPML = {
    'rta': 0.1535,
    'diplo_disagreement': 0.0756
}

def run_estimation():
    print("--- Démarrage de l'Estimation Polyads (Multivarié) ---")
    
    # -------------------------------------------------------------------------
    # 1. CHARGEMENT ET PRÉTRAITEMENT
    # -------------------------------------------------------------------------
    if not os.path.exists(DATA_PATH):
        print(f"❌ Fichier introuvable : {DATA_PATH}")
        return

    # Gestion de la lecture (Parquet ou CSV)
    try:
        df = pd.read_parquet(DATA_PATH)
    except:
        csv_path = DATA_PATH.replace('.parquet', '.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
        else:
            print("❌ Aucun fichier de données trouvé.")
            return

    # Vérification des colonnes requises
    required_cols = ['trade_flow', 'year', 'iso3num_o', 'iso3num_d', 'rta']
    if 'diplo_disagreement' in df.columns:
        required_cols.append('diplo_disagreement')
        features = ['rta', 'diplo_disagreement']
    else:
        print("⚠️ 'diplo_disagreement' absent. Estimation univariée (RTA seul).")
        features = ['rta']
        
    df = df.dropna(subset=required_cols).copy()

    # Scaling des flux (Essentiel pour la stabilité numérique)
    # Division par 1M pour éviter l'overflow int32
    print("⚖️  Scaling des flux commerciaux (divisé par 1 000 000)...")
    df['trade_flow'] = (df['trade_flow'] / 1_000_000)
    df['trade_flow'] = df['trade_flow'].fillna(0).astype(np.int32)
    
    print(f"Données prêtes : {len(df)} observations.")
    print(f"Variables explicatives (K={len(features)}) : {features}")

    # -------------------------------------------------------------------------
    # 2. ENCODAGE DES DIMENSIONS (N, N, T)
    # -------------------------------------------------------------------------
    print("🛠 Encodage des dimensions...")
    
    le_country = LabelEncoder()
    # Union des pays origine et destination pour un encodage cohérent
    all_countries = pd.concat([df['iso3num_o'], df['iso3num_d']]).unique()
    le_country.fit(all_countries)
    
    df['i'] = le_country.transform(df['iso3num_o'])
    df['j'] = le_country.transform(df['iso3num_d'])
    
    le_year = LabelEncoder()
    df['t'] = le_year.fit_transform(df['year'])
    
    n_countries = len(le_country.classes_)
    n_years = len(le_year.classes_)
    n_features = len(features)
    
    print(f"   -> Cube : {n_countries}x{n_countries}x{n_years} | Features : {n_features}")

    # -------------------------------------------------------------------------
    # 3. CONSTRUCTION DU TENSEUR X (Covariables)
    # -------------------------------------------------------------------------
    print("📦 Construction du Tenseur X...")
    
    # Initialisation du tenseur (N, N, T, K)
    X_tensor = np.zeros((n_countries, n_countries, n_years, n_features), dtype=np.float64)
    
    # Remplissage vectorisé
    idx_i = df['i'].values
    idx_j = df['j'].values
    idx_t = df['t'].values
    
    # Remplissage pour chaque variable explicative
    for k, feature_name in enumerate(features):
        X_tensor[idx_i, idx_j, idx_t, k] = df[feature_name].values
        
    print(f"   -> Tenseur construit. Taille en mémoire : {X_tensor.nbytes / 1024**2:.2f} MB")

    # -------------------------------------------------------------------------
    # 4. CONFIGURATION ET ESTIMATION
    # -------------------------------------------------------------------------
    # Initialisation de beta (vecteur de dimension K)
    beta_init = np.zeros(n_features) 
    
    estimator = PolyadEstimator(
        max_iter=100,
        tol=1e-6,
        max_n_polyads=int(5e7), # Augmenté pour la robustesse
        use_tqdm=True
    )
    
    print("⏳ Lancement de l'optimisation Polyads...")
    start_time = time.time()
    
    try:
        estimator.fit(
            df=df[['i', 'j', 't', 'trade_flow']], # DataFrame des flux
            indices=['i', 'j', 't'],              # Noms des colonnes indices
            values='trade_flow',                  # Nom de la colonne flux
            beta_init=beta_init,
            X=X_tensor                            # Le tenseur des régresseurs
        )
        
        duration = time.time() - start_time
        print(f"✅ Optimisation terminée en {duration:.2f} secondes.")
        
        # ---------------------------------------------------------------------
        # 5. RÉSULTATS ET COMPARAISON
        # ---------------------------------------------------------------------
        beta_hat = estimator.beta_
        
        # Tentative de récupération de la variance (si implémentée)
        std_errs = [np.nan] * n_features
        if hasattr(estimator, 'var_') and estimator.var_ is not None:
            try:
                std_errs = np.sqrt(np.diag(estimator.var_))
            except:
                pass

        print("\n" + "="*65)
        print(f"{'RÉSULTATS FINAUX : POLYADS vs PPML (Benchmark)':^65}")
        print("="*65)
        print(f"{'Variable':<20} | {'Polyads':<12} | {'SE':<10} | {'PPML (R)':<12}")
        print("-" * 75)
        
        for k, name in enumerate(features):
            b_poly = beta_hat[k]
            se_poly = std_errs[k]
            
            # Récupération du benchmark correspondant
            b_ppml = BENCHMARK_PPML.get(name, np.nan)
            
            diff = b_poly - b_ppml
            
            print(f"{name:<20} | {b_poly:<12.5f} | {se_poly:<10.4f} | {b_ppml:<12.5f} | {diff:<8.5f}")

        print("-" * 75)
        
        # Interprétation Rapide
        rta_idx = features.index('rta')
        effect_rta = (np.exp(beta_hat[rta_idx]) - 1) * 100
        print(f"\n💡 Impact Économique du RTA (Polyads) : +{effect_rta:.2f}%")
        
        if 'diplo_disagreement' in features:
            diplo_idx = features.index('diplo_disagreement')
            print(f"💡 Coefficient Politique (Diplo) : {beta_hat[diplo_idx]:.4f}")
            if beta_hat[diplo_idx] > 0:
                print("   (Note : Le coefficient positif confirme le paradoxe observé avec le PPML)")

    except Exception as e:
        print("\n❌ ERREUR CRITIQUE PENDANT L'ESTIMATION :")
        print(e)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_estimation()