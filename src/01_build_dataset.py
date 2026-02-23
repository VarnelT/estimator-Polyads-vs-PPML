import pandas as pd
import numpy as np
import os
import glob

# --- CONFIGURATION ---
YEARS = [2005, 2010, 2015]
BACI_PATH = "/home/onyxia/work/estimator-Polyads-vs-PPML/data/raw/baci_extracted"
GRAVITY_PATH = "/home/onyxia/work/estimator-Polyads-vs-PPML/data/raw/gravity_extracted" 
OUTPUT_PATH = "/home/onyxia/work/estimator-Polyads-vs-PPML/data/processed"

os.makedirs(OUTPUT_PATH, exist_ok=True)

def load_and_prep():
    """
    Construit le panel de données équilibré pour le Modèle de Gravité.
    Fusionne les flux commerciaux BACI avec les covariables CEPII Gravity.
    """
    print(f"--- Démarrage du pipeline de données (Années : {YEARS}) ---")

    # -------------------------------------------------------------------------
    # 1. CHARGEMENT ET AGRÉGATION DES FLUX COMMERCIAUX (BACI)
    # -------------------------------------------------------------------------
    list_dfs_trade = []
    
    for year in YEARS:
        # Localisation du fichier spécifique à l'année
        search_pattern = os.path.join(BACI_PATH, f"*{year}*.csv")
        candidates = glob.glob(search_pattern)
        # Sécurité : exclusion des fichiers Gravity s'ils sont dans le même dossier
        candidates = [f for f in candidates if "Gravity" not in f]
        
        if not candidates:
            print(f"[AVERTISSEMENT] Aucun fichier BACI trouvé pour l'année {year}")
            continue
            
        baci_file = candidates[0]
        print(f"Traitement : {os.path.basename(baci_file)}")
        
        try:
            # Chargement optimisé : uniquement les colonnes nécessaires
            df_chunk = pd.read_csv(baci_file, usecols=['t', 'i', 'j', 'v'])
            
            # Agrégation des flux (somme sur tous les produits) -> Commerce total bilatéral
            df_agg = df_chunk.groupby(['t', 'i', 'j'])['v'].sum().reset_index()
            
            # Standardisation des noms de colonnes
            df_agg.rename(columns={
                't': 'year', 
                'i': 'iso3num_o', 
                'j': 'iso3num_d', 
                'v': 'trade_flow'
            }, inplace=True)
            
            list_dfs_trade.append(df_agg)
            
        except Exception as e:
            print(f"[ERREUR] Échec du traitement pour {year} : {e}")

    if not list_dfs_trade:
        raise ValueError("Aucune donnée commerciale chargée. Vérifiez les chemins.")

    df_trade = pd.concat(list_dfs_trade)
    print(f"Données commerciales chargées : {len(df_trade)} observations.")

    # -------------------------------------------------------------------------
    # 2. CHARGEMENT DES COVARIABLES GRAVITY
    # -------------------------------------------------------------------------
    print("Traitement de la base Gravity...")
    grav_files = glob.glob(os.path.join(GRAVITY_PATH, "*Gravity*.csv"))
    
    if not grav_files:
        raise FileNotFoundError("Fichier Gravity introuvable.")
        
    df_grav = pd.read_csv(grav_files[0])
    
    # Filtre sur les années d'intérêt
    df_grav = df_grav[df_grav['year'].isin(YEARS)].copy()
    
    # Sélection stricte des variables requises
    # - Identifiants : year, iso3num (o/d), iso3 (o/d)
    # - Gestion des zéros structurels : distw (distance pondérée)
    # - Variables d'intérêt (Time-Varying) : fta_wto (RTA), diplo_disagreement (Distance Politique)
    
    cols_to_keep = [
        'year', 'iso3num_o', 'iso3num_d', 'iso3_o', 'iso3_d',
        'distw', 'fta_wto', 'diplo_disagreement'
    ]
    
    # Vérification de l'existence des colonnes (robustesse versions CEPII)
    existing_cols = [c for c in cols_to_keep if c in df_grav.columns]
    
    # Fallback pour le nom de la variable distance
    if 'distw' not in df_grav.columns and 'distw_harmonic' in df_grav.columns:
        existing_cols.append('distw_harmonic')
    
    df_grav = df_grav[existing_cols].copy()
    
    # Renommage pour cohérence
    rename_dict = {
        'fta_wto': 'rta',
        'distw_harmonic': 'distw'
    }
    df_grav.rename(columns=rename_dict, inplace=True)
    
    print(f"Données Gravity chargées : {len(df_grav)} observations.")

    # -------------------------------------------------------------------------
    # 3. FUSION ET NETTOYAGE
    # -------------------------------------------------------------------------
    print("Fusion des jeux de données...")
    
    # Left join sur Gravity pour préserver la structure des paires potentielles
    df_final = pd.merge(df_grav, df_trade, on=['year', 'iso3num_o', 'iso3num_d'], how='left')
    
    # Remplissage des flux manquants par 0 (Procédure standard Gravité)
    df_final['trade_flow'] = df_final['trade_flow'].fillna(0)
    
    # Suppression des observations avec régresseurs manquants
    # RTA et Distance sont obligatoires.
    # Diplo_disagreement est conservé si disponible, sinon la ligne est supprimée.
    required_vars = ['rta']
    if 'distw' in df_final.columns:
        required_vars.append('distw')
    if 'diplo_disagreement' in df_final.columns:
        required_vars.append('diplo_disagreement')
        
    initial_len = len(df_final)
    df_final.dropna(subset=required_vars, inplace=True)
    dropped_count = initial_len - len(df_final)
    
    print(f"Nettoyage terminé. {dropped_count} lignes supprimées (covariables manquantes).")
    print(f"Taille finale du dataset : {len(df_final)} observations.")

    # -------------------------------------------------------------------------
    # 4. EXPORT
    # -------------------------------------------------------------------------
    output_file = os.path.join(OUTPUT_PATH, "panel_total_trade.parquet")
    print(f"Sauvegarde dans {output_file}...")
    
    try:
        df_final.to_parquet(output_file, index=False)
    except ImportError:
        # Fallback CSV si pyarrow n'est pas installé
        csv_path = output_file.replace('.parquet', '.csv')
        print(f"Pyarrow non trouvé. Sauvegarde en CSV : {csv_path}")
        df_final.to_csv(csv_path, index=False)

    print("--- Pipeline terminé avec succès ---")

if __name__ == "__main__":
    load_and_prep()