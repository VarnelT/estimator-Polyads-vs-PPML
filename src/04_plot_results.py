import matplotlib.pyplot as plt
import numpy as np

# --- DONNÉES FINALES (RÉSULTATS OBTENUS) ---
# Résultats de ton script Polyads (Panel Complet)
polyads_beta = 0.11696
polyads_se   = 0.0470   # Standard Error calculé par Polyads

# Résultats du Benchmark PPML (Reference)
# (J'utilise 0.03 comme SE approximatif standard pour PPML sur ce type de données 
# si tu ne l'as pas noté, sinon remplace-le par la vraie valeur R)
ppml_beta    = 0.134566
ppml_se      = 0.03299   # Souvent plus faible car PPML sous-estime parfois la variance

# --- CONFIGURATION DU GRAPHIQUE ---
methods = ['PPML (Benchmark)', 'Polyads (Notre Modèle)']
betas   = [ppml_beta, polyads_beta]
errors  = [1.96 * ppml_se, 1.96 * polyads_se] # Intervalle de confiance à 95%

# Création de la figure
plt.figure(figsize=(8, 6))

# Couleurs : Bleu classique pour Benchmark, Rouge/Orange pour Polyads
colors = ['#1f77b4', '#ff7f0e']

# Plot des points et barres d'erreur
for i in range(len(methods)):
    plt.errorbar(x=betas[i], y=i, xerr=errors[i], fmt='o', 
                 markersize=10, capsize=5, color=colors[i], label=methods[i])

    # Annotation des valeurs
    plt.text(betas[i], i + 0.1, f"β = {betas[i]:.3f}", 
             ha='center', va='bottom', fontsize=10, fontweight='bold', color=colors[i])

# Esthétique
plt.yticks(range(len(methods)), methods, fontsize=12)
plt.xlabel("Effet Estimé de l'Accord Commercial (RTA)", fontsize=12)
#plt.title("Comparaison des Estimateurs : PPML vs Polyads\n(Intervalle de Confiance 95%)", fontsize=14)
plt.axvline(x=0, color='black', linestyle='--', alpha=0.3) # Ligne du zéro
plt.grid(axis='x', linestyle=':', alpha=0.6)

# Zone de "Recouvrement" (Validation)
# On grise la zone où les deux IC se croisent pour montrer la cohérence
overlap_min = max(betas[0] - errors[0], betas[1] - errors[1])
overlap_max = min(betas[0] + errors[0], betas[1] + errors[1])
if overlap_max > overlap_min:
    plt.axvspan(overlap_min, overlap_max, color='green', alpha=0.1, label='Zone de Consensus')

plt.legend(loc='lower left')

# Sauvegarde
output_path = "polyads_vs_ppml_final.png"
plt.tight_layout()
plt.savefig(output_path, dpi=300)
print(f"✅ Graphique sauvegardé sous : {output_path}")
plt.show()