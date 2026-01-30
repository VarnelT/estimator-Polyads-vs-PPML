import numpy as np
import pandas as pd
import time
import sys
import os
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# On ajoute le chemin pour trouver le module polyads
sys.path.append(os.getcwd())

try:
    from src.polyads.model import PolyadEstimator
except ImportError:
    try:
        from polyads.model import PolyadEstimator
    except ImportError:
        print("❌ PolyadEstimator introuvable.")
        sys.exit(1)

# --- CONFIGURATION ---
# Chemin dynamique vers les données
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
DATA_PATH = os.path.join(project_root, "data", "processed", "panel_total_trade.parquet")

# On garde le Top 40 pour la validation rapide
# (Tu pourras passer à 80 ou plus une fois que ce script aura affiché un résultat)
TOP_N_COUNTRIES = 40

def run_estimation():
    print("🚀 Démarrage de POLYADS (Version Propre & Rapide)...")
    
    # 1. CHARGEMENT
    if not os.path.exists(DATA_PATH):
        print(f"❌ Fichier introuvable : {DATA_PATH}")
        return

    try:
        df = pd.read_parquet(DATA_PATH)
    except:
        df = pd.read_csv(DATA_PATH.replace('.parquet', '.csv'))

    # Nettoyage de base
    df = df.dropna(subset=['distw', 'gdp_o', 'gdp_d', 'trade_flow', 'rta'])
    
    # 2. FILTRE TOP PAYS (Pour densifier le graphe)
    print(f"✂️  Sélection des {TOP_N_COUNTRIES} plus gros commerçants...")
    vol_par_pays = df.groupby('iso3_o')['trade_flow'].sum().sort_values(ascending=False)
    top_pays = vol_par_pays.head(TOP_N_COUNTRIES).index.tolist()
    
    # On ne garde que les flux au sein de ce groupe
    df = df[df['iso3_o'].isin(top_pays) & df['iso3_d'].isin(top_pays)].copy()
    print(f"   -> Reste : {len(df)} observations.")

    # 3. RENUMÉROTATION STRICTE & TYPAGE (C'est ce qui répare le bug)
    print("🛠 Renumérotation des indices (0 à N-1)...")
    
    # Pays
    le_pays = LabelEncoder()
    pays_presents = pd.concat([df['iso3_o'], df['iso3_d']]).unique()
    le_pays.fit(pays_presents)
    
    df['i'] = le_pays.transform(df['iso3_o']).astype(np.int32)
    df['j'] = le_pays.transform(df['iso3_d']).astype(np.int32)
    
    # Temps
    le_year = LabelEncoder()
    df['t'] = le_year.fit_transform(df['year']).astype(np.int32)
    
    n_i = len(le_pays.classes_)
    n_t = len(le_year.classes_)
    
    print(f"   -> Dimensions : {n_i} x {n_i} x {n_t}")
    
    # Conversion flux en int32 (Vital pour le C++)
    df['trade_flow'] = df['trade_flow'].astype(np.int32)
    
    # 4. TENSEUR X
    print("📦 Construction du Tenseur X...")
    X_tensor = np.zeros((n_i, n_i, n_t, 1), dtype=np.float64)
    
    idx_i = df['i'].values
    idx_j = df['j'].values
    idx_t = df['t'].values
    
    X_tensor[idx_i, idx_j, idx_t, 0] = df['rta'].values

    # 5. LANCEMENT
    print("⏳ Optimisation en cours...")
    
    # On laisse tqdm (barre de chargement) activé cette fois
    estimator = PolyadEstimator(
        max_iter=100,
        tol=1e-4,
        max_n_polyads=100000, # On augmente un peu car c'est performant maintenant
        use_tqdm=True
    )
    
    start_time = time.time()
    
    try:
        estimator.fit(
            df=df[['i', 'j', 't', 'trade_flow']],
            indices=['i', 'j', 't'],
            values='trade_flow',
            beta_init=np.array([0.0]),
            X=X_tensor
        )
        
        duration = time.time() - start_time
        
        # 6. RÉSULTATS
        beta_poly = estimator.beta_[0]
        # Benchmark monde (pour info)
        ppml_benchmark = 0.1346 
        
        print("\n" + "="*50)
        print("✅ RÉSULTAT FINAL")
        print("="*50)
        print(f"Polyads Beta (Top {TOP_N_COUNTRIES}) : {beta_poly:.5f}")
        print(f"Benchmark PPML (Monde) : {ppml_benchmark:.5f}")
        print(f"Temps de calcul        : {duration:.2f} s")
        
        effect = (np.exp(beta_poly) - 1) * 100
        print(f"\n💡 Effet estimé de l'accord : +{effect:.2f}%")

    except Exception as e:
        print("\n❌ Erreur :")
        print(e)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_estimation()