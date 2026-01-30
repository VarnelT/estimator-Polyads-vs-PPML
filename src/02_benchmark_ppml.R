# --- INSTALLATION AUTOMATIQUE DES PACKAGES ---
if(!require(fixest)) install.packages("fixest", repos="https://cloud.r-project.org")
if(!require(arrow)) install.packages("arrow", repos="https://cloud.r-project.org")
if(!require(data.table)) install.packages("data.table", repos="https://cloud.r-project.org")

library(fixest)
library(arrow)
library(data.table)

# --- 1. CHARGEMENT DES DONNÉES PANEL ---
cat("🚀 Chargement des données Panel (2005, 2010, 2015)...\n")
# Chemin vers ton nouveau fichier Panel
data_path <- "estimator-Polyads-vs-PPML/data/processed/panel_total_trade.parquet"

if (!file.exists(data_path)) {
  stop("❌ ERREUR : Le fichier 'panel_total_trade.parquet' est introuvable. Lance d'abord le script 01 !")
}

dt <- read_parquet(data_path)
setDT(dt) # Conversion en data.table pour la performance

cat("✅ Données chargées :", nrow(dt), "observations.\n")

# --- 2. PRÉPARATION ---
# Nettoyage de base : On s'assure que le RTA est propre
dt <- dt[!is.na(trade_flow) & !is.na(rta)]

# NOTE IMPORTANTE SUR LE MODÈLE B :
# Dans ce modèle structurel, on inclut des Effets Fixes Paires (ij).
# Par définition, la DISTANCE, la LANGUE, la FRONTIÈRE ne changent pas dans le temps pour une paire.
# Elles sont donc ABSORBÉES (colinéaires) par l'effet fixe. On ne les met pas dans la formule.
# De même, le PIB est absorbé par les effets fixes temps-pays.

# --- 3. L'ESTIMATION PPML (Modèle Structurel "Three-Way") ---
cat("🥊 Lancement du PPML Structurel (Modèle B)...\n")
cat("   Formule : Flux ~ RTA | Paire + Exportateur-Année + Importateur-Année\n")

# Syntaxe fixest :
# trade_flow ~ rta      <- La variable d'intérêt
# |                     <- Séparateur des effets fixes
# iso3_o^iso3_d         <- Effet Fixe PAIRE (Absorbe la distance, l'histoire, etc.)
# iso3_o^year           <- Effet Fixe Exportateur-Temps (Absorbe le PIB origine, l'offre, etc.)
# iso3_d^year           <- Effet Fixe Importateur-Temps (Absorbe le PIB destination, la demande, etc.)

model_ppml_B <- fepois(
  trade_flow ~ rta | iso3_o^iso3_d + iso3_o^year + iso3_d^year,
  data = dt,
  cluster = ~iso3_o^iso3_d # On cluster les erreurs standard par paire (robuste à l'autocorrélation)
)

# --- 4. RÉSULTATS ---
print(model_ppml_B)

cat("\n📊 --- ANALYSE FINALE (BENCHMARK) ---\n")
coeff_rta <- coef(model_ppml_B)["rta"]
effect_pct <- (exp(coeff_rta) - 1) * 100

cat(sprintf("Coefficient RTA (log-odds) : %.4f\n", coeff_rta))
cat(sprintf("👉 IMPACT ÉCONOMIQUE ESTIMÉ : +%.2f%% de commerce grâce à l'accord.\n", effect_pct))
cat("----------------------------------------------------------\n")
cat("C'est CE chiffre précis que Polyads doit retrouver en utilisant\n")
cat("la dimension temporelle (t) et les interactions tenseurs.\n")