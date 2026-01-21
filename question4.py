

import sys
import os


# Configuration de l'encodage UTF-8
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Ajouter le répertoire parent (racine du projet) au PYTHONPATH
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from src.visualization import plot_likelihood_vs_beta1

# ============================================================================
# DONNÉES D'ENTRAÎNEMENT
# ============================================================================
x_train = np.array([
    0.09291784, 0.46809093, 0.93089486, 0.67612654, 0.73441752, 0.86847339,
    0.49873225, 0.51083168, 0.18343972, 0.99380898, 0.27840809, 0.38028817,
    0.12055708, 0.56715537, 0.92005746, 0.77072270, 0.85278176, 0.05315950,
    0.87168699, 0.58858043
])

y_train = np.array([
    0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1
])

# ============================================================================
# PARAMÈTRES FIXES DU RÉSEAU
# ============================================================================
# On fixe tous les paramètres sauf beta1 qu'on va faire varier
beta0 = np.array([0.3, -1.0, -0.5])    # Biais de la couche cachée
omega0 = np.array([-1.0, 1.8, 0.65])   # Poids de la couche cachée
omega1 = np.array([-24.0, -8.0, 50.0]) # Poids de la couche de sortie

# ============================================================================
# CONFIGURATION DE L'EXPLORATION DE BETA1
# ============================================================================
print("=" * 70)
print("QUESTION 4: OPTIMISATION DE beta_1")
print("=" * 70)
print("\nParametres fixes:")
print(f"  beta_0 (biais couche cachee)    : {beta0}")
print(f"  omega_0 (poids couche cachee)   : {omega0}")
print(f"  omega_1 (poids couche de sortie): {omega1}")
print(f"\nNombre d'exemples d'entrainement: {len(x_train)}")

print("\n" + "-" * 70)
print("EXPLORATION DE beta_1")
print("-" * 70)
print("Plage de beta_1: [-30, 30]")
print("Nombre de points: 500")
print("Objectif: Trouver beta_1 optimal qui:")
print("  1. Maximise le Likelihood L(theta)")
print("  2. Minimise la Negative Log-Likelihood NLL(theta)")
print("-" * 70)

# ============================================================================
# EXÉCUTION DE L'OPTIMISATION ET VISUALISATION
# ============================================================================
print("\nCalcul en cours...")
print("(Ce calcul peut prendre quelques secondes)")

# La fonction plot_likelihood_vs_beta1:
# - Fait varier beta1 de -30 à 30 avec 500 points
# - Calcule L et NLL pour chaque beta1
# - Trace les deux courbes
# - Identifie et affiche les valeurs optimales
# - Retourne (beta1_max_L, beta1_min_NLL)

beta1_max_L, beta1_min_NLL = plot_likelihood_vs_beta1(
    x_train, 
    y_train, 
    beta0, 
    omega0, 
    omega1,
    beta1_range=(-30, 30, 500)  # (min, max, nombre de points)
)

# ============================================================================
# VÉRIFICATION FINALE
# ============================================================================
print("\n" + "=" * 70)
print("VERIFICATION: MAXIMISER L <=> MINIMISER NLL")
print("=" * 70)

# Calculer la différence entre les deux valeurs optimales
difference = abs(beta1_max_L - beta1_min_NLL)

print(f"beta_1 qui maximise L  : {beta1_max_L:.6f}")
print(f"beta_1 qui minimise NLL: {beta1_min_NLL:.6f}")
print(f"Difference absolue     : {difference:.10f}")

# Tolérance pour considérer que les valeurs sont égales
tolerance = 1e-6

if difference < tolerance:
    print(f"\n[OK] VERIFICATION REUSSIE (difference < {tolerance})")
    print("     Les deux approches donnent le meme beta_1 optimal!")
else:
    print(f"\n[!] Difference detectee: {difference:.6f}")
    print("    (Peut etre du a la resolution de la grille de recherche)")

# ============================================================================
# CONCLUSION
# ============================================================================
print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
print("\n>> RESULTATS EMPIRIQUES:")
print(f"   beta_1 optimal = {beta1_max_L:.4f}")
print("\n>> THEORIE VERIFIEE:")
print("   Maximiser le Likelihood <=> Minimiser la Negative Log-Likelihood")
print("\n>> EXPLICATION:")
print("   - NLL = -log(L)")
print("   - La fonction log est monotone croissante")
print("   - Donc: argmax L(theta) = argmin NLL(theta)")
print("\n>> AVANTAGES DE NLL:")
print("   1. Evite le sous-flux numerique (pas de tres petites valeurs)")
print("   2. Transforme produit -> somme (plus facile a optimiser)")
print("   3. Utilise dans tous les frameworks deep learning modernes")
print("=" * 70)



