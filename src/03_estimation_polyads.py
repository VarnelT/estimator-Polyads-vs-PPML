import os

# --- 🛑 CORRECTIF ANTI-BLOCAGE (A mettre AVANT d'importer numpy) ---
# On force Python à n'utiliser qu'un seul thread pour éviter les conflits
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
# -------------------------------------------------------------------

import numpy as np
import pandas as pd
import time
import sys
from sklearn.preprocessing import LabelEncoder

# On désactive tqdm pour voir les vrais logs (parfois la barre de chargement cache les erreurs)
# from tqdm import tqdm 

sys.path.append(os.getcwd())

try:
    from src.polyads.model import PolyadEstimator
except ImportError:
    try:
        from polyads.model import PolyadEstimator
    except ImportError:
        print("❌ ERREUR IMPORT : PolyadEstimator introuvable.")
        sys.exit(1)

# --- CONFIGURATION TEST ATOMIQUE ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
DATA_PATH = os.path.join(project_root, "data", "processed", "panel_total_trade.parquet")

# ON RESTE SUR LE TEST MINUSCULE POUR VALIDER
TOP_N_COUNTRIES = 10 

def run_estimation():
    print("🚀 Démarrage du Test Polyads (Mode Monocoeur Sécurisé)...")
    
    if not os.path.exists(DATA_PATH):
        print(f"❌ Fichier introuvable : {DATA_PATH}")
        return

    # 1. Chargement & Nettoyage
    try:
        df = pd.read_parquet(DATA_PATH)
    except:
        df = pd.read_csv(DATA_PATH.replace('.parquet', '.csv'))
    
    df['rta'] = df['rta'].fillna(0).astype(int)
    df['trade_flow'] = df['trade_flow'].fillna(0).astype(int)
    df = df.dropna(subset=['distw', 'gdp_o', 'gdp_d'])

    # 2. Filtre Drastique (Top 10)
    print(f"✂️  Filtrage sur les {TOP_N_COUNTRIES} plus gros pays...")
    total_trade = df.groupby('iso3_o')['trade_flow'].sum().sort_values(ascending=False)
    top_countries = total_trade.head(TOP_N_COUNTRIES).index.tolist()
    df = df[df['iso3_o'].isin(top_countries) & df['iso3_d'].isin(top_countries)].copy()
    
    print(f"   -> Reste {len(df)} observations.")

    # 3. Encodage
    le_pays = LabelEncoder()
    all_countries = pd.concat([df['iso3_o'], df['iso3_d']]).unique()
    le_pays.fit(all_countries)
    df['i'] = le_pays.transform(df['iso3_o'])
    df['j'] = le_pays.transform(df['iso3_d'])
    
    le_year = LabelEncoder()
    df['t'] = le_year.fit_transform(df['year'])
    
    n_i = len(le_pays.classes_)
    n_t = len(le_year.classes_)

    # 4. Tenseur X
    print("📦 Construction Tenseur...")
    X_tensor = np.zeros((n_i, n_i, n_t, 1), dtype=np.float64)
    idx_i = df['i'].values.astype(int)
    idx_j = df['j'].values.astype(int)
    idx_t = df['t'].values.astype(int)
    X_tensor[idx_i, idx_j, idx_t, 0] = df['rta'].values

    # 5. Estimation
    print("⏳ Lancement Optimisation (Cela peut prendre 10-20 sec pour compiler)...")
    
    # On met use_tqdm=False pour éviter les bugs d'affichage
    estimator = PolyadEstimator(
        max_iter=10,
        tol=1e-3, 
        max_n_polyads=1000, 
        use_tqdm=False 
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
        print("\n" + "="*40)
        print(f"✅ SUCCÈS ! Le blocage est levé.")
        print(f"Beta estimé : {estimator.beta_[0]:.5f}")
        print(f"Temps       : {duration:.2f} s")
        print("="*40)

    except Exception as e:
        print("\n❌ ERREUR :")
        print(e)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_estimation()