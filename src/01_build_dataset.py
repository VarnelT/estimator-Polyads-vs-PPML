import pandas as pd
import numpy as np
import os
import glob
import country_converter as coco  # La nouvelle librairie magique

# --- CONFIGURATION ---
BACI_PATH = "data/raw/baci_extracted"
GRAVITY_PATH = "data/raw/gravity_extracted"
OUTPUT_PATH = "data/processed"

# On choisit le secteur "61" (Vêtements en maille)
TARGET_HS2 = '61' 

def load_and_prep():
    print("🚀 Démarrage du Data Engineering (Version Sans Fichier Codes)...")
    
    # 1. Identifier les fichiers de données
    try:
        baci_file = glob.glob(os.path.join(BACI_PATH, "BACI_HS92_Y2015*.csv"))[0]
        gravity_file = glob.glob(os.path.join(GRAVITY_PATH, "Gravity_*.csv"))[0]
    except IndexError:
        print("❌ ERREUR : Impossible de trouver BACI ou Gravity.")
        return

    print(f"📄 Lecture de BACI : {os.path.basename(baci_file)}")
    
    # 2. Charger et Filtrer BACI (Optimisation Mémoire)
    chunks = []
    print(f"⏳ Filtrage du secteur HS {TARGET_HS2} en cours...")
    
    # On lit le CSV par morceaux
    for chunk in pd.read_csv(baci_file, chunksize=1000000):
        chunk['k'] = chunk['k'].astype(str)
        filtered = chunk[chunk['k'].str.startswith(TARGET_HS2)].copy()
        chunks.append(filtered)
    
    df_baci = pd.concat(chunks)
    print(f"✅ BACI chargé : {len(df_baci)} lignes de flux existants.")

    # 3. TRADUCTION AUTOMATIQUE DES CODES PAYS (via country_converter)
    print("🌍 Traduction des codes pays (Numérique -> ISO3)...")
    
    # On récupère tous les codes uniques présents dans BACI (Origine 'i' et Destination 'j')
    unique_codes = set(df_baci['i'].unique()) | set(df_baci['j'].unique())
    
    # On utilise la librairie pour convertir (src='un' car BACI utilise les codes Nations Unies)
    cc = coco.CountryConverter()
    # On crée un dictionnaire : {251: 'FRA', 842: 'USA', ...}
    # map_iso = cc.convert(names=list(unique_codes), src='numeric', to='ISO3', not_found=None)
    # Note: BACI utilise des codes numériques standard (souvent UN M49)
    
    # Astuce : On convertit en série pour faire le mapping
    iso_map = cc.convert(names=list(unique_codes), src='un', to='ISO3', not_found=None)
    
    # Si la conversion renvoie une liste (cas normal), on zippe
    if isinstance(iso_map, list):
        code_dict = dict(zip(list(unique_codes), iso_map))
    else:
        # Cas où il n'y aurait qu'un seul pays (peu probable)
        code_dict = {list(unique_codes)[0]: iso_map}

    # On applique la traduction
    df_baci['iso_o'] = df_baci['i'].map(code_dict)
    df_baci['iso_d'] = df_baci['j'].map(code_dict)
    
    # On nettoie ceux qui n'ont pas été trouvés (ex: 'not_found')
    df_baci = df_baci[df_baci['iso_o'] != 'not_found']
    df_baci = df_baci[df_baci['iso_d'] != 'not_found']

    print(f"✅ Traduction terminée. Lignes valides : {len(df_baci)}")

    # 4. Charger Gravity (Juste 2015)
    print("📄 Lecture de Gravity...")
    df_grav = pd.read_csv(gravity_file)
    df_grav = df_grav[df_grav['year'] == 2015]
    
    # --- CORRECTION DES NOMS DE COLONNES ---
    # On sélectionne les noms exacts que tu as trouvés
    cols_grav_raw = ['iso3_o', 'iso3_d', 'distw_harmonic', 'fta_wto', 'gdp_o', 'gdp_d', 'contig', 'comlang_off']
    
    # On vérifie qu'elles sont toutes là (sécurité)
    missing = [c for c in cols_grav_raw if c not in df_grav.columns]
    if missing:
        print(f"❌ ERREUR CRITIQUE : Il manque encore ces colonnes : {missing}")
        return

    # On ne garde que ça
    df_grav = df_grav[cols_grav_raw]

    # On renomme pour que ça matche avec la suite du script ('distw' et 'rta')
    df_grav.rename(columns={
        'distw_harmonic': 'distw',
        'fta_wto': 'rta'
    }, inplace=True)
    
    print(f"✅ Gravity chargée et corrigée : {len(df_grav)} paires.")

    # 5. CRÉATION DU SQUELETTE (Les Zéros)
    print("Création du Squelette (Injection des Zéros)...")
    
    unique_products = df_baci['k'].unique()
    valid_pairs = df_grav.dropna(subset=['gdp_o', 'gdp_d'])[['iso3_o', 'iso3_d']]
    
    # Produit Cartésien optimisé
    df_products = pd.DataFrame({'k': unique_products})
    df_products['key'] = 1
    valid_pairs['key'] = 1
    
    print(f"   -> {len(unique_products)} produits x {len(valid_pairs)} paires pays")
    
    df_skeleton = pd.merge(valid_pairs, df_products, on='key').drop('key', axis=1)
    
    print(f"   -> Squelette prêt : {len(df_skeleton)} lignes théoriques.")

    # 6. MERGE FINAL
    print("🔗 Fusion finale...")
    
    df_final = pd.merge(df_skeleton, df_baci, 
                        left_on=['iso3_o', 'iso3_d', 'k'], 
                        right_on=['iso_o', 'iso_d', 'k'], 
                        how='left')
    
    # Remplir les NaNs par 0
    df_final['v'] = df_final['v'].fillna(0)
    df_final['q'] = df_final['q'].fillna(0)
    
    # Ajouter Gravity info
    df_final = pd.merge(df_final, df_grav, on=['iso3_o', 'iso3_d'], how='left')
    
    # Nettoyage final
    cols_to_keep = ['iso3_o', 'iso3_d', 'k', 'v', 'distw', 'rta', 'gdp_o', 'gdp_d', 'contig', 'comlang_off']
    df_final = df_final[cols_to_keep]
    df_final.rename(columns={'v': 'trade_flow', 'k': 'product_code'}, inplace=True)

    # 7. SAUVEGARDE
    print("💾 Sauvegarde...")
    output_file = os.path.join(OUTPUT_PATH, "trade_data_2015_textile_sparse.parquet")
    df_final.to_parquet(output_file, index=False)
    
    print(f"🎉 SUCCÈS ! Fichier généré : {output_file}")

if __name__ == "__main__":
    load_and_prep()