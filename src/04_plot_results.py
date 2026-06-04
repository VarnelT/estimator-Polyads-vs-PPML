import matplotlib.pyplot as plt
import numpy as np

# --- DONNÉES DES ESTIMATIONS (Vecteur de dimension 2) ---
variables = ['Accord Commercial (RTA)', 'Désaccord Diplomatique']

# PPML (Benchmark R - fixest)
betas_ppml = [0.15354, 0.07560]
ses_ppml = [0.03294, 0.02638]
ci_ppml = [1.96 * s for s in ses_ppml]

# Polyads (Résultats Python)
betas_poly = [0.12209, 0.07572]
ses_poly = [0.04660, 0.04370]
ci_poly = [1.96 * s for s in ses_poly]

# --- CONFIGURATION DU GRAPHIQUE ---
y_pos = np.arange(len(variables))
offset = 0.15  # Décalage vertical pour séparer les deux modèles

fig, ax = plt.subplots(figsize=(12, 7), dpi=300)

# Tracé des coefficients PPML (en bleu)
ax.errorbar(x=betas_ppml, y=y_pos + offset, xerr=ci_ppml, 
            fmt='o', markersize=10, capsize=6, linewidth=2, 
            color='#1f77b4', label='PPML (Benchmark)', linestyle='None')

# Tracé des coefficients Polyads (en orange)
ax.errorbar(x=betas_poly, y=y_pos - offset, xerr=ci_poly, 
            fmt='s', markersize=10, capsize=6, linewidth=2, 
            color='#ff7f0e', label='Polyads (Notre Modèle)', linestyle='None')

# Ajout des valeurs numériques au-dessus/en-dessous des points
for i in range(len(variables)):
    # Valeurs PPML
    ax.text(betas_ppml[i], y_pos[i] + offset + 0.08, f"β = {betas_ppml[i]:.4f}", 
            ha='center', va='bottom', fontsize=10, fontweight='bold', color='#1f77b4')
    # Valeurs Polyads
    ax.text(betas_poly[i], y_pos[i] - offset - 0.22, f"β = {betas_poly[i]:.4f}", 
            ha='center', va='bottom', fontsize=10, fontweight='bold', color='#ff7f0e')

# --- ESTHÉTIQUE ET FINITIONS ---
ax.axvline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.4) # Ligne de l'effet nul
ax.grid(axis='x', linestyle='--', alpha=0.5)

ax.set_yticks(y_pos)
ax.set_yticklabels(variables, fontsize=12, fontweight='bold')
ax.set_xlabel("Coefficient estimé (Intervalles de confiance à 95%)", fontsize=12)
ax.set_title("Comparaison des Estimateurs : RTA et Diplomatie", fontsize=14, pad=20)
ax.set_xlim(-0.05, 0.25) # Ajustement de l'axe pour la visibilité

ax.legend(loc='upper right', fontsize=11, frameon=True)

plt.tight_layout()
plt.savefig("/home/onyxia/work/estimator-Polyads-vs-PPML/results/graphique_final_multivarie.png")