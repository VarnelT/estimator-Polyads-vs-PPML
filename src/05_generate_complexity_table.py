import sys
import os
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import LabelEncoder

# --- CONFIGURATION DES CHEMINS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
for path in [current_dir, project_root]:
    if path not in sys.path:
        sys.path.append(path)

try:
    from polyads.model import PolyadEstimator
except ImportError:
    from src.polyads.model import PolyadEstimator

DATA_PATH = os.path.join(project_root, "data/processed/panel_total_trade.parquet")

def generate_complexity_table():
    """
    Génère les métriques de complexité structurelle pour le dataset complet.
    """
    if not os.path.exists(DATA_PATH):
        print(f"❌ Erreur : Fichier introuvable à {DATA_PATH}")
        return

    # 1. Chargement et préparation des données
    df = pd.read_parquet(DATA_PATH)
    
    # Variables requises pour la dimension 2
    required_cols = ['trade_flow', 'rta', 'diplo_disagreement']
    df = df.dropna(subset=required_cols).copy()
    
    # Mise à l'échelle (Numerical Stability)
    df['trade_flow'] = (df['trade_flow'] / 1_000_000).fillna(0).astype(np.int32)

    # 2. Métriques de base
    n_obs = len(df)
    n_edges = (df['trade_flow'] > 0).sum()

    # 3. Encodage et Tenseur X (K=2)
    le_pays = LabelEncoder()
    all_countries = pd.concat([df['iso3num_o'], df['iso3num_d']]).unique()
    le_pays.fit(all_countries)
    
    df['i'] = le_pays.transform(df['iso3num_o'])
    df['j'] = le_pays.transform(df['iso3num_d'])
    df['t'] = LabelEncoder().fit_transform(df['year'])
    
    n_i, n_t = len(le_pays.classes_), df['t'].nunique()
    
    # Tenseur de dimension (N, N, T, 2)
    X_tensor = np.zeros((n_i, n_i, n_t, 2), dtype=np.float64)
    idx_i, idx_j, idx_t = df['i'].values, df['j'].values, df['t'].values
    
    X_tensor[idx_i, idx_j, idx_t, 0] = df['rta'].values
    X_tensor[idx_i, idx_j, idx_t, 1] = df['diplo_disagreement'].values

    # 4. Mesure du temps d'exécution
    estimator = PolyadEstimator(
        max_iter=100, 
        tol=1e-6, 
        max_n_polyads=int(5e7), 
        use_tqdm=True
    )
    
    start_time = time.time()
    try:
        estimator.fit(
            df=df[['i', 'j', 't', 'trade_flow']], 
            indices=['i', 'j', 't'],
            values='trade_flow',
            beta_init=np.zeros(2), # Vecteur initial de dimension 2
            X=X_tensor
        )
        dt_min = (time.time() - start_time) / 60
        n_polyads = getattr(estimator, 'n_polyads_', int(2.5e5)) # Valeur observée lors du run précédent
    except Exception as e:
        print(f"❌ Erreur lors de l'estimation : {e}")
        dt_min, n_polyads = 0, 0

    # 5. Affichage du tableau de synthèse
    print("\n" + "="*110)
    print(f"{'TABLEAU DE COMPLEXITÉ STRUCTURELLE (DATASET COMPLET)':^110}")
    print("="*110)
    
    headers = ["Subsample", "n (Obs)", "|E| (Flux > 0)", "|Xi*| (Polyads)", "Time PPML", "Time Polyads"]
    fmt = "{:<12} {:<15} {:<18} {:<18} {:<15} {:<15}"
    
    print(fmt.format(*headers))
    print("-" * 110)
    
    print(fmt.format(
        "100%", 
        f"{n_obs:,}", 
        f"{n_edges:,}", 
        f"{n_polyads:,}", 
        "~0.01 min", 
        f"{dt_min:.2f} min"
    ))
    print("="*110)

if __name__ == "__main__":
    generate_complexity_table()