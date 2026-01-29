import pandas as pd
import numpy as np
import os
import glob

# --- CONFIGURATION ---
YEARS = [2005, 2010, 2015]  # Nos 3 années pivots
BACI_PATH = "data/raw/baci_extracted"
# Note : Gravity est souvent extrait au même endroit ou à la racine de raw
GRAVITY_PATH = "data/raw/gravity_extracted" 
OUTPUT_PATH = "data/processed"

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
        
        # Optimisation : On ne lit que les colonnes utiles
        # t=Year, i=Exporter(num), j=Importer(num), v=Value
        # On ignore 'k' (produit) car on va tout sommer
        try:
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
    # 2. TRAITEMENT GRAVITY
    # =========================================================================
    print("📄 Lecture de Gravity...")
    # On cherche le fichier Gravity
    grav_files = glob.glob(os.path.join(GRAVITY_PATH, "*Gravity*.csv"))
    if not grav_files:
        print("❌ CRITIQUE : Fichier Gravity introuvable.")
        return
        
    df_grav = pd.read_csv(grav_files[0])
    
    # Filtre sur nos années
    df_grav = df_grav[df_grav['year'].isin(YEARS)].copy()
    
    # Sélection et Renommage
    # On garde les codes ISO3 string (ex: FRA) pour la lisibilité, mais on merge sur les NUM (250)
    cols_grav = ['year', 'iso3num_o', 'iso3num_d', 'iso3_o', 'iso3_d', 
                 'distw_harmonic', 'fta_wto', 'gdp_o', 'gdp_d', 'contig', 'comlang_off']
    
    # Petit filtre de sécurité si des colonnes manquent
    cols_to_keep = [c for c in cols_grav if c in df_grav.columns]
    df_grav = df_grav[cols_to_keep]
    
    df_grav.rename(columns={'distw_harmonic': 'distw', 'fta_wto': 'rta'}, inplace=True)
    
    print(f"✅ Gravity chargée : {len(df_grav)} lignes.")

    # =========================================================================
    # 3. FUSION ET NETTOYAGE
    # =========================================================================
    print("🔗 Fusion Commerce + Gravité (Left Join sur Gravity)...")
    
    # On merge sur Année + Code Numérique Exportateur + Code Numérique Importateur
    # C'est beaucoup plus sûr que les codes lettres qui changent parfois
    df_final = pd.merge(df_grav, df_trade, on=['year', 'iso3num_o', 'iso3num_d'], how='left')
    
    # Injection des Zéros (Sparsity)
    # Si on a une ligne Gravity mais pas de flux BACI -> C'est un zéro
    n_zeros = df_final['trade_flow'].isna().sum()
    print(f"   -> Injection de {n_zeros} flux nuls (zéros commerciaux).")
    df_final['trade_flow'] = df_final['trade_flow'].fillna(0)
    
    # Nettoyage Macro
    df_final = df_final.dropna(subset=['distw', 'gdp_o', 'gdp_d'])
    
    # =========================================================================
    # 4. SAUVEGARDE
    # =========================================================================
    output_file = os.path.join(OUTPUT_PATH, "panel_total_trade.parquet")
    print(f"💾 Sauvegarde dans {output_file}...")
    
    # On sauvegarde en Parquet (rapide) ou CSV si pyarrow manque
    try:
        df_final.to_parquet(output_file, index=False)
        print("🎉 SUCCÈS ! Dataset Panel prêt pour Polyads.")
    except ImportError:
        print("⚠️ Pyarrow absent, sauvegarde en CSV...")
        df_final.to_csv(output_file.replace('.parquet', '.csv'), index=False)

if __name__ == "__main__":
    load_and_prep()