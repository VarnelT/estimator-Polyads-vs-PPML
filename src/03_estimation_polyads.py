import numpy as np
import pandas as pd
import time
import sys
import os
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

sys.path.append(os.getcwd())

try:
    from src.polyads.model import PolyadEstimator
except ImportError:
    try:
        from polyads.model import PolyadEstimator
    except ImportError:
        print("❌ ERREUR IMPORT : Impossible de charger PolyadEstimator.")
        sys.exit(1)

# --- CONFIGURATION ---
# On pointe vers le PANEL construit à l'étape précédente
DATA_PATH = "data/processed/panel_total_trade.parquet"

def run_estimation():
    print("🚀 Chargement du PANEL (2005, 2010, 2015)...")
    if not os.path.exists(DATA_PATH):
        print(f"❌ Fichier introuvable : {DATA_PATH}")
        return

    try:
        df = pd.read_parquet(DATA_PATH)
    except:
        df = pd.read_csv(DATA_PATH.replace('.parquet', '.csv'))
    
    # =========================================================================
    # 1. NETTOYAGE & SCALING (PARTIE MODIFIÉE)
    # =========================================================================
    
    # Nettoyage RTA (0 ou 1)
    df['rta'] = df['rta'].fillna(0).astype(int)
    
    # Filtre des données manquantes (PIB, Distance...)
    df = df.dropna(subset=['distw', 'gdp_o', 'gdp_d', 'trade_flow'])

    print("⚖️  Scaling des flux (Numerical Stability)...")
    
    # --- LA MODIFICATION EST ICI ---
    # Division par 1 000 000 pour :
    # 1. Éviter l'overflow des entiers 32-bit (Max ~2 Milliards)
    # 2. Stabiliser les gradients de l'algorithme (éviter l'explosion)
    df['trade_flow'] = (df['trade_flow'] / 1_000_000)
    
    # Conversion stricte en int32 (Format obligatoire pour le C++ de Polyads)
    df['trade_flow'] = df['trade_flow'].fillna(0).astype(np.int32)
    
    print(f"✅ Données chargées et scalées : {len(df)} observations.")
    print(f"   -> Flux Max (en millions) : {df['trade_flow'].max()} (Safe < 2 Mrds)")

    # =========================================================================
    # 2. ENCODAGE DES DIMENSIONS (i, j, t)
    print("🛠 Encodage des dimensions (Pays, Année)...")
    
    le_pays = LabelEncoder()
    # On apprend les codes sur l'ensemble des pays (Origine + Destination)
    all_countries = pd.concat([df['iso3_o'], df['iso3_d']]).unique()
    le_pays.fit(all_countries)
    
    df['i'] = le_pays.transform(df['iso3_o'])
    df['j'] = le_pays.transform(df['iso3_d'])
    
    le_year = LabelEncoder()
    df['t'] = le_year.fit_transform(df['year'])
    
    n_i = len(le_pays.classes_) # Nombre de pays
    n_t = len(le_year.classes_) # Nombre d'années (3)
    
    print(f"   -> Cube Panel : {n_i} Pays x {n_i} Pays x {n_t} Années")

    # 3. CONSTRUCTION DU TENSEUR X
    # CRUCIAL : En Panel structurel, les variables constantes (Distance, Langue) sautent !
    # On ne garde que ce qui varie dans le temps : RTA.
    print("📦 Construction du Tenseur 3D (Variable RTA uniquement)...")
    
    # Dimension 4 = 1 seule variable (RTA)
    X_tensor = np.zeros((n_i, n_i, n_t, 1), dtype=np.float64)
    
    # Remplissage Vectorisé
    indices_i = df['i'].values
    indices_j = df['j'].values
    indices_t = df['t'].values
    
    # On remplit avec le RTA
    X_tensor[indices_i, indices_j, indices_t, 0] = df['rta'].values
    
    print(f"   -> Tenseur prêt. Taille : {X_tensor.nbytes / 1024**2:.2f} MB")

    # 4. CONFIGURATION DE L'ESTIMATEUR
    beta_init = np.array([0.0]) # On cherche 1 seul coefficient (RTA)
    
    estimator = PolyadEstimator(
        max_iter=100,
        tol=1e-6,
        max_n_polyads=int(2e7), # On autorise beaucoup de polyads
        use_tqdm=True
    )
    
    # 5. ESTIMATION
    print("⏳ Démarrage de l'optimisation Polyads...")
    start_time = time.time()
    
    try:
        estimator.fit(
            df=df[['i', 'j', 't', 'trade_flow']],
            indices=['i', 'j', 't'], # Dimensions Panel
            values='trade_flow',
            beta_init=beta_init,
            X=X_tensor
        )
        
        duration = time.time() - start_time
        print(f"✅ TERMINÉ en {duration:.2f} s")
        
        # 6. RÉSULTATS
        beta_hat = estimator.beta_[0]
        
        # Erreur standard (si dispo)
        se = np.nan
        if hasattr(estimator, 'var_') and estimator.var_ is not None:
             try:
                se = np.sqrt(np.diag(estimator.var_))[0]
             except: pass
        
        # Benchmark PPML (Calculé en R précédemment)
        PPML_BENCHMARK = 0.1346 
        
        print("\n" + "="*50)
        print("RÉSULTATS FINAUX : POLYADS PANEL vs PPML")
        print("="*50)
        
        print(f"{'Variable':<10} | {'Coef Polyads':<15} | {'Std.Err':<10} | {'Cible PPML':<10}")
        print("-" * 55)
        print(f"{'RTA':<10} | {beta_hat:<15.5f} | {se:<10.4f} | {PPML_BENCHMARK:<10.4f}")
        print("-" * 55)
        
        # Calcul de l'écart
        diff = abs(beta_hat - PPML_BENCHMARK)
        ecart_pct = (diff / PPML_BENCHMARK) * 100
        
        print(f"\n📊 DIAGNOSTIC :")
        print(f"Écart avec le benchmark structurel : {diff:.5f} ({ecart_pct:.2f}%)")
        
        effect_pct = (np.exp(beta_hat) - 1) * 100
        print(f"\n💡 INTERPRÉTATION ÉCO :")
        print(f"Polyads estime que l'accord augmente le commerce de +{effect_pct:.2f}%")
        print("(Contrôlé par effets fixes Paires, Origine-Temps, Destination-Temps)")

    except Exception as e:
        print("\n❌ ERREUR :")
        print(e)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_estimation()