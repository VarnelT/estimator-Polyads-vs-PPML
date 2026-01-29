import numpy as np
import pandas as pd
import time
import sys
import os

# Ajout du chemin racine pour être sûr de trouver le module src
sys.path.append(os.getcwd())

try:
    # On essaie l'import comme indiqué dans le README
    from src.polyads.model import PolyadEstimator
except ImportError:
    try:
        # Fallback si installé comme package système
        from polyads.model import PolyadEstimator
    except ImportError:
        print("❌ ERREUR IMPORT : Impossible de charger PolyadEstimator.")
        print("Vérifie que tu es bien à la racine du projet.")
        sys.exit(1)

# --- CONFIGURATION ---
DATA_PATH = "data/processed/trade_data_2015_textile_sparse.parquet"

def run_estimation():
    print("🚀 Chargement des données pour Polyads...")
    df = pd.read_parquet(DATA_PATH)
    
    # 1. NETTOYAGE ET PRÉPARATION
    # On vire les lignes où les variables explicatives sont nulles ou infinies
    df = df[df['distw'] > 0].copy()
    df['ln_dist'] = np.log(df['distw'])
    df = df.dropna(subset=['rta', 'ln_dist', 'trade_flow'])
    
    # On s'assure que les flux sont des entiers (Count Data)
    df['trade_flow'] = df['trade_flow'].astype(int)
    
    print(f"✅ Données chargées : {len(df)} observations.")

    # 2. PRÉPARATION DE LA MATRICE X
    # Selon la doc, on peut passer X comme une matrice (N_obs, P_features)
    # si elle correspond aux lignes du DataFrame.
    print("Construction de la matrice X (Covariables)...")
    
    # Ordre des variables : [RTA, Ln_Dist]
    X_matrix = df[['rta', 'ln_dist']].to_numpy(dtype=np.float64)
    
    # 3. INITIALISATION
    # Initialisation des bêtas à 0
    beta_init = np.zeros(X_matrix.shape[1]) 
    
    print("🥊 Initialisation de l'estimateur Polyads...")
    estimator = PolyadEstimator(
        max_iter=100,           # Standard
        tol=1e-4,               # Tolérance standard
        max_n_polyads=int(1e7), # On limite un peu pour pas exploser la RAM au début
        use_tqdm=True           # Barre de progression (cool sur Onyxia)
    )
    
    # 4. LANCEMENT DU FIT
    print("⏳ Démarrage de l'estimation...")
    print(f"   (Dimensions : {len(df)} lignes x {X_matrix.shape[1]} variables)")
    
    start_time = time.time()
    
    try:
        estimator.fit(
            df=df,
            indices=['iso3_o', 'iso3_d', 'product_code'], # Les colonnes qui définissent les noeuds
            values='trade_flow',     # La colonne cible (Y)
            beta_init=beta_init,
            X=X_matrix
        )
        
        end_time = time.time()
        print(f"✅ TERMINÉ en {end_time - start_time:.2f} secondes !")
        
        # 5. DIAGNOSTICS & RÉSULTATS
        if not estimator.converged_:
            print("\n⚠️ ATTENTION : L'algorithme n'a pas convergé !")
        else:
            print("\n🎉 CONVERGENCE RÉUSSIE !")
            
        print("\n" + "="*40)
        print("RÉSULTATS FINAUX (POLYADS)")
        print("="*40)
        
        # Affichage propre via la méthode summary du package
        estimator.summary(alpha=0.05)
        
        # Comparaison manuelle si summary() est trop verbeux
        betas = estimator.beta_
        se = np.sqrt(np.diag(estimator.var_))
        
        print("\n--- TABLEAU COMPARATIF RAPIDE ---")
        print(f"{'Variable':<15} | {'Coef (Polyads)':<15} | {'Std.Err':<10}")
        print("-" * 45)
        print(f"{'RTA':<15} | {betas[0]:<15.4f} | {se[0]:<10.4f}")
        print(f"{'Ln_Dist':<15} | {betas[1]:<15.4f} | {se[1]:<10.4f}")
        print("-" * 45)
        
        # Interprétation immédiate
        rta_effect = (np.exp(betas[0]) - 1) * 100
        print(f"\n💡 Interprétation Éco :")
        print(f"Selon Polyads, un accord commercial augmente le commerce de {rta_effect:.2f}%")
        print(f"(Rappel PPML : +141%. Si ce chiffre est plus bas, tu as prouvé le biais !)")

    except Exception as e:
        print("\n❌ ERREUR CRITIQUE PENDANT LE FIT :")
        print(e)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_estimation()