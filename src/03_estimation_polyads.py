import numpy as np
import pandas as pd
import time
import sys
import os
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm # On l'importe explicitement maintenant qu'elle est installée

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
DATA_PATH = "data/processed/trade_data_2015_textile_sparse.parquet"

def run_estimation():
    print("🚀 Chargement des données pour Polyads...")
    df = pd.read_parquet(DATA_PATH)
    
    # 1. NETTOYAGE
    # On enlève les distances nulles (log(0) impossible)
    df = df[df['distw'] > 0].copy()
    df['ln_dist'] = np.log(df['distw'])
    # On enlève les lignes incomplètes
    df = df.dropna(subset=['rta', 'ln_dist', 'trade_flow'])
    df['trade_flow'] = df['trade_flow'].astype(int) 
    
    print(f"✅ Données brutes : {len(df)} observations.")

    # 2. ENCODAGE (Label Encoding)
    print("🛠 Transformation des codes en entiers...")
    
    le_pays = LabelEncoder()
    # On apprend les codes sur l'ensemble (Origine + Destination) pour être cohérent
    tous_les_pays = pd.concat([df['iso3_o'], df['iso3_d']]).unique()
    le_pays.fit(tous_les_pays)
    
    df['i'] = le_pays.transform(df['iso3_o'])
    df['j'] = le_pays.transform(df['iso3_d'])
    
    le_prod = LabelEncoder()
    df['k'] = le_prod.fit_transform(df['product_code'])
    
    # Récupération des dimensions du cube
    n_i = len(le_pays.classes_) # Nombre d'exportateurs
    n_j = len(le_pays.classes_) # Nombre d'importateurs
    n_k = len(le_prod.classes_) # Nombre de produits
    
    print(f"   -> Dimensions du Cube : {n_i} x {n_j} x {n_k}")

    # 3. CONSTRUCTION DU TENSEUR 3D (C'est ICI que ça change !)
    # Au lieu d'une liste, on crée un cube vide (n_i, n_j, n_k, 2 variables)
    print("📦 Construction du Tenseur 3D (X)...")
    
    # 2 variables explicatives : RTA (index 0) et Ln_Dist (index 1)
    # On initialise tout à 0.
    X_tensor = np.zeros((n_i, n_j, n_k, 2), dtype=np.float64)
    
    # Remplissage Vectorisé (Rapide) ⚡
    # On utilise les indices i, j, k pour placer les valeurs au bon endroit dans le cube
    # Note : Comme on a injecté les zéros à l'étape 1, le dataframe couvre la majorité du cube.
    indices_i = df['i'].values
    indices_j = df['j'].values
    indices_k = df['k'].values
    
    # Variable 1 : RTA
    X_tensor[indices_i, indices_j, indices_k, 0] = df['rta'].values
    # Variable 2 : Ln_Dist
    X_tensor[indices_i, indices_j, indices_k, 1] = df['ln_dist'].values
    
    print(f"   -> Tenseur prêt. Taille en mémoire : {X_tensor.nbytes / 1024**2:.2f} MB")

    # 4. PRÉPARATION DU MODÈLE
    beta_init = np.zeros(2) # 2 coefficients à trouver
    
    # Création d'un DF simplifié pour le fit (contient juste les indices et le flux)
    df_model = df[['i', 'j', 'k', 'trade_flow']].copy()
    
    print("🥊 Initialisation de l'estimateur Polyads...")
    estimator = PolyadEstimator(
        max_iter=100,
        tol=1e-4,
        max_n_polyads=int(1e7), 
        use_tqdm=True
    )
    
    # 5. LANCEMENT DU FIT
    print("⏳ Démarrage de l'estimation...")
    start_time = time.time()
    
    try:
        estimator.fit(
            df=df_model,      # DataFrame avec i, j, k, trade_flow
            indices=['i', 'j', 'k'], 
            values='trade_flow',
            beta_init=beta_init,
            X=X_tensor        # ON PASSE LE TENSEUR 3D ICI !
        )
        
        end_time = time.time()
        print(f"✅ TERMINÉ en {end_time - start_time:.2f} secondes !")
        
        # 6. RÉSULTATS
        if not estimator.converged_:
            print("\n⚠️ ATTENTION : L'algorithme n'a pas convergé !")
        else:
            print("\n🎉 CONVERGENCE RÉUSSIE !")
            
        print("\n" + "="*40)
        print("RÉSULTATS DU MATCH : POLYADS vs PPML")
        print("="*40)
        
        betas = estimator.beta_
        
        # Gestion sécurisée des erreurs standards
        if hasattr(estimator, 'var_') and estimator.var_ is not None:
             try:
                se = np.sqrt(np.diag(estimator.var_))
             except:
                se = [np.nan, np.nan]
        else:
             se = [np.nan, np.nan]
        
        print(f"{'Variable':<15} | {'Coef (Polyads)':<15} | {'Std.Err':<10}")
        print("-" * 45)
        print(f"{'RTA':<15} | {betas[0]:<15.4f} | {se[0]:<10.4f}")
        print(f"{'Ln_Dist':<15} | {betas[1]:<15.4f} | {se[1]:<10.4f}")
        print("-" * 45)
        
        rta_effect_poly = (np.exp(betas[0]) - 1) * 100
        print(f"\n💡 ANALYSE FINALE :")
        print(f"Effet RTA estimé par Polyads : +{rta_effect_poly:.2f}%")
        print(f"(Compare ça avec le +141% du PPML !)")

    except Exception as e:
        print("\n❌ ERREUR PENDANT LE CALCUL :")
        print(e)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_estimation()