import pandas as pd
import numpy as np
import os
import glob

# --- CONFIGURATION ---
YEARS = [2005, 2010, 2015]  # Nos 3 années pivots
BACI_PATH = "/home/onyxia/work/estimator-Polyads-vs-PPML/data/raw/baci_extracted"
GRAVITY_PATH = "/home/onyxia/work/estimator-Polyads-vs-PPML/data/raw/gravity_extracted" 
OUTPUT_PATH = "/home/onyxia/work/estimator-Polyads-vs-PPML/data/processed"

os.makedirs(OUTPUT_PATH, exist_ok=True)

def load_and_prep():
    print(f"🚀 Démarrage du Data Engineering (Panel Total Trade {YEARS})...")
    
    list_dfs_trade = []

    # =========================================================================
    # 1. TRAITEMENT BACI (Boucle sur les années)
    # =========================================================================
    for year in YEARS:
        # On cherche le fichier correspondant à l'année (ex: *2005*.csv)
        # On exclut "Gravity" de la recherche pour ne pas se tromper
        search_pattern = os.path.join(BACI_PATH, f"*{year}*.csv")
        candidates = glob.glob(search_pattern)
        candidates = [f for f in candidates if "Gravity" not in f]
        
        if not candidates:
            print(f"⚠️  Pas de fichier trouvé pour {year} dans {BACI_PATH}")
            continue
            
        baci_file = candidates[0]
        print(f"📄 Lecture de BACI {year} : {os.path.basename(baci_file)}")
        
        try:
            # Optimisation : On ne lit que les colonnes utiles
            # t=Year, i=Exporter(num), j=Importer(num), v=Value
            df_chunk = pd.read_csv(baci_file, usecols=['t', 'i', 'j', 'v'])
            
            # AGRÉGATION : Somme de tous les produits -> 1 ligne par paire de pays
            print(f"   ⏳ Agrégation du commerce total pour {year}...")
            df_agg = df_chunk.groupby(['t', 'i', 'j'])['v'].sum().reset_index()
            
            # Renommage standard
            df_agg.rename(columns={
                't': 'year', 
                'i': 'iso3num_o', 
                'j': 'iso3num_d', 
                'v': 'trade_flow'
            }, inplace=True)
            
            list_dfs_trade.append(df_agg)
            
        except Exception as e:
            print(f"❌ Erreur lecture {year}: {e}")

    if not list_dfs_trade:
        print("❌ CRITIQUE : Aucun donnée de commerce chargée.")
        return

    # Fusion verticale de toutes les années
    df_trade = pd.concat(list_dfs_trade)
    print(f"✅ Panel Commerce construit : {len(df_trade)} lignes.")

    # =========================================================================
    # 2. TRAITEMENT GRAVITY (AJOUT DES VARIABLES DE CONTRÔLE)
    # =========================================================================
    print("📄 Lecture de Gravity...")
    grav_files = glob.glob(os.path.join(GRAVITY_PATH, "*Gravity*.csv"))
    if not grav_files:
        print("❌ CRITIQUE : Fichier Gravity introuvable.")
        return
        
    # On charge le fichier Gravity
    # Note: Assure-toi que c'est bien le fichier 'Gravity_csv_Vxxxx.csv' complet
    df_grav = pd.read_csv(grav_files[0])
    
    # Filtre sur nos années
    df_grav = df_grav[df_grav['year'].isin(YEARS)].copy()
    
    # --- MODIFICATION ICI : AJOUT DES VARIABLES DE CONTRÔLE ---
    # On ajoute 'colony' qui manquait, et on s'assure d'avoir les standards
    cols_grav = [
        'year', 'iso3num_o', 'iso3num_d',   # Clés de jointure
        'iso3_o', 'iso3_d',                 # Codes ISO3 lettres (pour lisibilité)
        'distw',                            # Distance pondérée (Standard Gravity)
        'fta_wto',                          # Notre variable d'intérêt (RTA)
        'contig',                           # Frontière commune
        'comlang_off',                      # Langue officielle commune
        'colony',                           # Lien colonial (AJOUTÉ)
        'gdp_o', 'gdp_d'                    # PIB (pour info ou scaling)
    ]
    
    # Petit filtre de sécurité : on ne prend que les colonnes qui existent vraiment dans le fichier
    # (Parfois 'distw' s'appelle 'distw_harmonic' selon la version de CEPII)
    available_cols = df_grav.columns.tolist()
    
    # Gestion de la distance : distw ou distw_harmonic ?
    if 'distw' not in available_cols and 'distw_harmonic' in available_cols:
        df_grav.rename(columns={'distw_harmonic': 'distw'}, inplace=True)
    
    cols_to_keep = [c for c in cols_grav if c in df_grav.columns]
    df_grav = df_grav[cols_to_keep]
    
    # Renommage pour standardiser
    rename_dict = {
        'fta_wto': 'rta',           # Plus clair pour l'économétrie
        'distw_harmonic': 'distw'   # Au cas où
    }
    df_grav.rename(columns=rename_dict, inplace=True)
    
    print(f"✅ Gravity chargée avec contrôles : {len(df_grav)} lignes.")

    # =========================================================================
    # 3. FUSION ET NETTOYAGE
    # =========================================================================
    print("🔗 Fusion Commerce + Gravité (Left Join sur Gravity)...")
    
    # On merge sur Année + Code Numérique Exportateur + Code Numérique Importateur
    df_final = pd.merge(df_grav, df_trade, on=['year', 'iso3num_o', 'iso3num_d'], how='left')
    
    # Injection des Zéros (Sparsity)
    n_zeros = df_final['trade_flow'].isna().sum()
    print(f"   -> Injection de {n_zeros} flux nuls (zéros commerciaux).")
    df_final['trade_flow'] = df_final['trade_flow'].fillna(0)
    
    # Nettoyage des NaNs dans les variables explicatives
    # On ne peut pas faire de régression si la distance ou le PIB est vide
    vars_to_check = ['distw', 'gdp_o', 'gdp_d', 'rta', 'contig', 'colony']
    # On ne check que celles qui sont présentes
    vars_to_check = [v for v in vars_to_check if v in df_final.columns]
    
    before_drop = len(df_final)
    df_final = df_final.dropna(subset=vars_to_check)
    print(f"🧹 Nettoyage NaNs : {before_drop} -> {len(df_final)} observations.")
    
    # =========================================================================
    # 4. SAUVEGARDE
    # =========================================================================
    output_file = os.path.join(OUTPUT_PATH, "panel_total_trade.parquet")
    print(f"💾 Sauvegarde dans {output_file}...")
    
    try:
        df_final.to_parquet(output_file, index=False)
        print("🎉 SUCCÈS ! Dataset Panel prêt pour Polyads (avec contrôles).")
    except ImportError:
        print("⚠️ Pyarrow absent, sauvegarde en CSV...")
        df_final.to_csv(output_file.replace('.parquet', '.csv'), index=False)

if __name__ == "__main__":
    load_and_prep()