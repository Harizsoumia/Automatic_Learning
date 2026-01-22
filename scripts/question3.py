


import sys
import os

# Configuration de l'encodage UTF-8
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Ajouter le répertoire parent (racine du projet) au PYTHONPATH
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from src.metrics import calculate_likelihood, calculate_negative_log_likelihood

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
# PARAMÈTRES DU RÉSEAU DE NEURONES
# ============================================================================
# Couche cachée (3 neurones avec activation ReLU)
beta0 = np.array([0.3, -1.0, -0.5])    # Biais de la couche cachée
omega0 = np.array([-1.0, 1.8, 0.65])   # Poids de la couche cachée

# Couche de sortie (1 neurone linéaire)
beta1 = 2.6                             # Biais de la couche de sortie
omega1 = np.array([-24.0, -8.0, 50.0]) # Poids de la couche de sortie

# ============================================================================
# AFFICHAGE DES PARAMÈTRES
# ============================================================================
print("=" * 70)
print("QUESTION 3: CALCUL DE LA NEGATIVE LOG-LIKELIHOOD")
print("=" * 70)
print("\nParametres du reseau:")
print(f"  beta_0 (biais couche cachee)    : {beta0}")
print(f"  omega_0 (poids couche cachee)   : {omega0}")
print(f"  beta_1 (biais couche de sortie) : {beta1}")
print(f"  omega_1 (poids couche de sortie): {omega1}")
print(f"\nNombre d'exemples d'entrainement: {len(x_train)}")

# ============================================================================
# CALCUL DU LIKELIHOOD (Vraisemblance)
# ============================================================================
# L(theta) = produit P(yi | xi, theta)
# C'est le produit des probabilités individuelles selon Bernoulli
likelihood = calculate_likelihood(x_train, y_train, beta0, omega0, beta1, omega1)

print("\n" + "-" * 70)
print("RESULTAT 1: LIKELIHOOD (Vraisemblance)")
print("-" * 70)
print(f"L(theta) = {likelihood:.6e}")
print("\nInterpretation:")
print("  - Le Likelihood mesure la probabilite conjointe des donnees")
print("  - Valeur tres petite car c'est un produit de 20 probabilites")
print("  - Probleme: risque de sous-flux numerique (underflow)")

# ============================================================================
# CALCUL DE LA NEGATIVE LOG-LIKELIHOOD
# ============================================================================
# NLL = -log L(theta) = -somme[yi*log(p_i) + (1-yi)*log(1-p_i)]
# Equivalent a Binary Cross-Entropy (BCE)
nll = calculate_negative_log_likelihood(x_train, y_train, beta0, omega0, beta1, omega1)

print("\n" + "-" * 70)
print("RESULTAT 2: NEGATIVE LOG-LIKELIHOOD (NLL)")
print("-" * 70)
print(f"NLL(theta) = {nll:.6f}")
print("\nInterpretation:")
print("  - NLL = -log(Likelihood)")
print("  - Transforme le produit en somme -> plus stable numeriquement")
print("  - NLL est equivalent a la Binary Cross-Entropy (BCE)")
print("  - On minimise NLL au lieu de maximiser Likelihood")

# ============================================================================
# VÉRIFICATION: RELATION ENTRE LIKELIHOOD ET NLL
# ============================================================================
print("\n" + "-" * 70)
print("VERIFICATION: RELATION LIKELIHOOD <-> NLL")
print("-" * 70)

# Si NLL = -log(L), alors L = exp(-NLL)
likelihood_from_nll = np.exp(-nll)

print(f"Likelihood calcule directement  : {likelihood:.6e}")
print(f"Likelihood depuis NLL (e^(-NLL)): {likelihood_from_nll:.6e}")
print(f"Difference absolue              : {abs(likelihood - likelihood_from_nll):.6e}")

# Vérification
if np.isclose(likelihood, likelihood_from_nll, rtol=1e-10):
    print("\n[OK] VERIFICATION REUSSIE: NLL = -log(Likelihood)")
else:
    print("\n[!] ATTENTION: Difference detectee (probablement due a la precision numerique)")

# ============================================================================
# RÉSUMÉ FINAL
# ============================================================================
print("\n" + "=" * 70)
print("RESUME")
print("=" * 70)
print(f"Likelihood (L)              : {likelihood:.6e}")
print(f"Negative Log-Likelihood (NLL): {nll:.6f}")
print(f"Log-Likelihood (log L)      : {np.log(likelihood):.6f}")
print(f"Verification: -log(L) = NLL : {-np.log(likelihood):.6f}")
print("=" * 70)

print("\n>> CONCLUSION:")
print("   La Negative Log-Likelihood est calculee avec succes.")
print("   Elle represente la Binary Cross-Entropy pour ce probleme.")
print("   Dans la Question 4, nous allons optimiser beta_1 pour minimiser NLL.")
print("=" * 70)
