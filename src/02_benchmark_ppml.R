library(fixest)
library(arrow)
library(data.table)

# --- 1. CONFIGURATION ET CHARGEMENT ---

# Chemin d'accès relatif standardisé
DATA_PATH <- "estimator-Polyads-vs-PPML/data/processed/panel_total_trade.parquet"

# Fallback pour environnement local ou racine différente
if (!file.exists(DATA_PATH)) {
  DATA_PATH <- "data/processed/panel_total_trade.parquet"
}

if (!file.exists(DATA_PATH)) {
  stop("Erreur critique : Fichier de données introuvable. Veuillez exécuter le script 01 au préalable.")
}

cat("Chargement du panel de données...\n")
dt <- read_parquet(DATA_PATH)
setDT(dt) # Conversion data.table

cat(sprintf("Observations chargées : %s\n", format(nrow(dt), big.mark = ",")))

# --- 2. DÉFINITION DU MODÈLE ---

# Identification des covariables disponibles
# Le modèle inclut RTA par défaut, et 'diplo_disagreement' si disponible dans le parquet
covariates <- c("rta")
if ("diplo_disagreement" %in% names(dt)) {
  covariates <- c(covariates, "diplo_disagreement")
}

# Construction dynamique de la formule
# Structure : Y ~ X | Pair_FE + Exporter_Time_FE + Importer_Time_FE
rhs_formula <- paste(covariates, collapse = " + ")
fml_structurelle <- as.formula(paste(
  "trade_flow ~", rhs_formula, "| iso3_o^iso3_d + iso3_o^year + iso3_d^year"
))

cat("\n--- Spécification du Modèle ---\n")
print(fml_structurelle)
cat("Note : Les variables dyadiques invariantes (distance, frontières) sont absorbées par les effets fixes paires.\n")

# --- 3. ESTIMATION (PPML) ---

cat("\nExécution de l'estimation PPML (fepois)...\n")

model_ppml <- fepois(
  fml = fml_structurelle,
  data = dt,
  cluster = ~iso3_o^iso3_d # Clustering des erreurs standard au niveau de la paire
)

# --- 4. RÉSULTATS ET INTERPRÉTATION ---

cat("\n--- Résultats de l'estimation ---\n")
print(model_ppml)

# Extraction et affichage de l'impact économique pour le RTA
if ("rta" %in% names(coef(model_ppml))) {
  beta_rta <- coef(model_ppml)["rta"]
  se_rta <- se(model_ppml)["rta"]
  effect_pct <- (exp(beta_rta) - 1) * 100
  
  cat("\n--- Analyse du coefficient RTA ---\n")
  cat(sprintf("Coefficient (Beta) : %.4f (SE: %.4f)\n", beta_rta, se_rta))
  cat(sprintf("Effet marginal estimé : +%.2f%% sur les flux commerciaux.\n", effect_pct))
}

cat("\nFin du script.\n")