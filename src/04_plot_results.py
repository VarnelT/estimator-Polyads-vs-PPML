import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. DONNÉES (À VERIFIER AVEC TES RESULTATS)
# ==========================================

# A. PPML (Benchmark R fixest - Cluster Paire)
beta_ppml = 0.13457
se_ppml   = 0.03299
ci_ppml   = 1.96 * se_ppml  # Marge d'erreur à 95%

# B. POLYADS (Ton résultat Python)
# (Mets ici les valeurs exactes de ton dernier run)
beta_poly = 0.11700 
se_poly   = 0.04700 
ci_poly   = 1.96 * se_poly

# ==========================================
# 2. CONFIGURATION DU GRAPHIQUE
# ==========================================
# On prépare les données pour l'axe Y (0 pour PPML, 1 pour Polyads)
y_pos = [0, 1]
betas = [beta_ppml, beta_poly]
errors = [ci_ppml, ci_poly]
labels = ['PPML (Benchmark)', 'Polyads (Notre Modèle)']
colors = ['#1f77b4', '#ff7f0e'] # Bleu (Standard), Orange (Polyads)

# Création de la figure
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

# ==========================================
# 3. ZONE DE CONSENSUS (Le Rectangle Vert)
# ==========================================
# On cherche l'intersection des deux intervalles de confiance
low_overlap  = max(beta_ppml - ci_ppml, beta_poly - ci_poly)
high_overlap = min(beta_ppml + ci_ppml, beta_poly + ci_poly)

# On dessine la zone verte
if high_overlap > low_overlap:
    ax.axvspan(low_overlap, high_overlap, color='green', alpha=0.1, label='Zone de Consensus')
    # Lignes pointillées verticales pour délimiter la zone
    ax.axvline(low_overlap, color='green', linestyle=':', alpha=0.3)
    ax.axvline(high_overlap, color='green', linestyle=':', alpha=0.3)

# ==========================================
# 4. TRACÉ DES ESTIMATEURS (ERROR BARS)
# ==========================================
for i in range(2):
    # Barre d'erreur
    ax.errorbar(x=betas[i], y=y_pos[i], xerr=errors[i], 
                fmt='o',             # 'o' pour un gros point
                markersize=12,       # Taille du point
                capsize=8,           # Taille des "chapeaux" aux bouts de la barre
                linewidth=2,         # Epaisseur du trait
                color=colors[i],     # Couleur
                label=labels[i])     # Pour la légende
    
    # Annotation du texte (Valeur de Beta) au-dessus du point
    # On décale le texte un peu vers le haut (+0.15)
    ax.text(x=betas[i], y=y_pos[i] + 0.15, s=f"β = {betas[i]:.3f}",
            ha='center', va='bottom', 
            fontsize=12, fontweight='bold', color=colors[i])

# ==========================================
# 5. ESTHÉTIQUE ET FINITIONS
# ==========================================

# Ligne du Zéro (Effet Nul)
ax.axvline(0, color='gray', linestyle='-.', linewidth=1.5, alpha=0.5)

# Grille verticale légère
ax.grid(axis='x', linestyle=':', alpha=0.6)

# Axe Y : On met les noms des modèles
ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=12)

# Axe X : Titre et limites
ax.set_xlabel("Effet Estimé de l'Accord Commercial (RTA)", fontsize=13)
# On fixe les limites pour que ce soit joli (un peu de marge à gauche et à droite)
ax.set_xlim(-0.02, 0.25)

# Titre global (Optionnel, sinon tu le mets dans LaTeX)
# ax.set_title("Comparaison des Estimateurs : Robustesse Structurelle", fontsize=14)

# Légende en bas à gauche
ax.legend(loc='lower left', fontsize=10, frameon=True)

# Mise en page serrée
plt.tight_layout()

# Sauvegarde
output_file = "graphique_final_polyads.png"
plt.savefig(output_file)
print(f"✅ Graphique généré : {output_file}")
plt.show()