# --- INSTALLATION AUTOMATIQUE DES PACKAGES ---
if(!require(fixest)) install.packages("fixest", repos="https://cloud.r-project.org")
if(!require(arrow)) install.packages("arrow", repos="https://cloud.r-project.org")
if(!require(data.table)) install.packages("data.table", repos="https://cloud.r-project.org")

library(fixest)
library(arrow)
library(data.table)

# --- 1. CHARGEMENT DES DONNÉES ---
cat("🚀 Chargement des données Parquet...\n")
# Adapte le nom du fichier si tu as changé la date ou le secteur
data_path <- "estimator-Polyads-vs-PPML/data/processed/trade_data_2015_textile_sparse.parquet"
dt <- read_parquet(data_path)

cat("✅ Données chargées :", nrow(dt), "lignes.\n")

# --- 2. PRÉPARATION ---
# On s'assure que les variables clés sont bien numériques et on gère les logs
# Pour la distance et le PIB, on prend le log (standard en gravité)
dt$ln_dist <- log(dt$distw)
dt$ln_gdp_o <- log(dt$gdp_o)
dt$ln_gdp_d <- log(dt$gdp_d)

# Petit check de propreté : virer les lignes avec des NaNs dans les régresseurs
dt <- na.omit(dt, cols = c("trade_flow", "ln_dist", "ln_gdp_o", "ln_gdp_d", "rta"))

# --- 3. L'ESTIMATION PPML (Le Standard) ---
cat("🥊 Lancement du PPML (fixest)...\n")
# Formule : Flux ~ Variables Gravité | Effets Fixes
# On met des Effets Fixes : Origine, Destination et Produit
model_ppml <- fepois(trade_flow ~ rta + ln_dist + ln_gdp_o + ln_gdp_d + contig + comlang_off | iso3_o + iso3_d + product_code, 
                     data = dt, 
                     cluster = ~iso3_o + iso3_d) # Clustering des erreurs standard (robuste)

# --- 4. RÉSULTATS ---
print(model_ppml)

cat("\n📊 --- ANALYSE RAPIDE ---\n")
coeff_rta <- coef(model_ppml)["rta"]
cat(sprintf("L'effet d'un accord commercial (RTA) est estimé à : +%.2f%%\n", (exp(coeff_rta)-1)*100))
cat("(C'est ce chiffre qu'on va essayer de challenger avec Polyads !)\n")
