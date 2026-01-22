"""
Question 2: Calcul du Likelihood des données d'entraînement

Ce script:
1. Crée le réseau de neurones avec les paramètres donnés
2. Calcule les probabilités prédites lambda(x) pour chaque point
3. Calcule le Likelihood selon la loi de Bernoulli
4. Affiche les résultats détaillés
5. Sauvegarde les résultats dans un fichier texte

Formule du Likelihood:
L(theta) = produit[lambda_i^yi * (1-lambda_i)^(1-yi)] pour i=1..N

Auteur: Soumia Hariz
"""

import sys
import os

# Configuration de l'encodage UTF-8
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Ajouter le répertoire parent (racine du projet) au PYTHONPATH
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from src.model import NeuralNetwork
from src.metrics import calculate_likelihood

# ============================================================================
# DONNÉES D'ENTRAÎNEMENT
# ============================================================================
x_train = np.array([
    0.09291784, 0.46809093, 0.93089486, 0.67612654, 
    0.73441752, 0.86847339, 0.49873225, 0.51083168, 
    0.18343972, 0.99380898, 0.27840809, 0.38028817,
    0.12055708, 0.56715537, 0.92005746, 0.77072270, 
    0.85278176, 0.05315950, 0.87168699, 0.58858043
])

y_train = np.array([
    0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 
    0, 1, 0, 1, 1, 0, 1, 0, 1, 1
])

# ============================================================================
# PARAMÈTRES DU RÉSEAU DE NEURONES
# ============================================================================
beta_0 = np.array([0.3, -1.0, -0.5])      # Biais couche cachée
omega_0 = np.array([-1.0, 1.8, 0.65])     # Poids couche cachée
beta_1 = 2.6                               # Biais couche de sortie
omega_1 = np.array([-24.0, -8.0, 50.0])   # Poids couche de sortie

# ============================================================================
# CALCUL DU LIKELIHOOD
# ============================================================================
print("=" * 70)
print("QUESTION 2: CALCUL DU LIKELIHOOD")
print("=" * 70)

print("\nParametres du reseau:")
print(f"  beta_0  : {beta_0}")
print(f"  omega_0 : {omega_0}")
print(f"  beta_1  : {beta_1}")
print(f"  omega_1 : {omega_1}")
print(f"\nNombre d'exemples d'entrainement: {len(x_train)}")

# Créer le modèle
model = NeuralNetwork(beta_0, omega_0, beta_1, omega_1)
print("\n[OK] Modele cree avec succes")

# Calculer les probabilités lambda(x) = P(y=1|x) pour chaque point
print("\n" + "-" * 70)
print("CALCUL DES PROBABILITES PREDITES")
print("-" * 70)

lambda_pred = model.predict_proba(x_train)

print(f"Probabilites calculees pour {len(lambda_pred)} points")
print(f"  Minimum : {np.min(lambda_pred):.6f}")
print(f"  Maximum : {np.max(lambda_pred):.6f}")
print(f"  Moyenne : {np.mean(lambda_pred):.6f}")

# Calculer le Likelihood en utilisant la fonction de metrics.py
print("\n" + "-" * 70)
print("CALCUL DU LIKELIHOOD")
print("-" * 70)

likelihood = calculate_likelihood(x_train, y_train, beta_0, omega_0, beta_1, omega_1)

print(f"\nLikelihood L(theta) = {likelihood:.15e}")
print(f"Log-Likelihood      = {np.log(likelihood):.15f}")

# ============================================================================
# AFFICHAGE DÉTAILLÉ: TABLEAU DES CONTRIBUTIONS
# ============================================================================
print("\n" + "-" * 70)
print("TABLEAU DETAILLE DES CONTRIBUTIONS AU LIKELIHOOD")
print("-" * 70)

# Calculer les contributions individuelles
contributions = (lambda_pred ** y_train) * ((1 - lambda_pred) ** (1 - y_train))

print(f"\n{'i':<4} | {'x_i':<12} | {'y_i':<6} | {'lambda_i':<12} | {'Contribution':<15}")
print("-" * 70)

for i in range(len(x_train)):
    print(f"{i:<4} | {x_train[i]:<12.6f} | {y_train[i]:<6} | "
          f"{lambda_pred[i]:<12.6f} | {contributions[i]:<15.6e}")

# Vérification: le produit de toutes les contributions doit égaler le likelihood
likelihood_verif = np.prod(contributions)
print("\n" + "-" * 70)
print("VERIFICATION")
print("-" * 70)
print(f"Produit des contributions : {likelihood_verif:.15e}")
print(f"Likelihood calcule        : {likelihood:.15e}")
print(f"Difference                : {abs(likelihood - likelihood_verif):.15e}")

if np.isclose(likelihood, likelihood_verif):
    print("\n[OK] VERIFICATION REUSSIE")
else:
    print("\n[!] ATTENTION: Difference detectee")

# ============================================================================
# INTERPRETATION DU LIKELIHOOD
# ============================================================================
print("\n" + "=" * 70)
print("INTERPRETATION DU LIKELIHOOD")
print("=" * 70)

print(f"""
Le Likelihood L(theta) = {likelihood:.6e} represente la probabilite 
conjointe d'observer les donnees y_train sachant les entrees x_train 
et les parametres theta du modele.

Formule de Bernoulli pour chaque observation i:
  P(y_i | x_i, theta) = lambda_i^(y_i) * (1 - lambda_i)^(1 - y_i)

Likelihood total:
  L(theta) = produit[P(y_i | x_i, theta)] pour i = 1 a {len(x_train)}

Cette valeur est tres petite car:
  - C'est un produit de {len(x_train)} probabilites
  - Chaque probabilite est entre 0 et 1
  - Le produit diminue exponentiellement

C'est pourquoi on prefere utiliser le Log-Likelihood:
  log L(theta) = {np.log(likelihood):.6f}
""")

# ============================================================================
# SAUVEGARDE DES RÉSULTATS
# ============================================================================
print("\n" + "-" * 70)
print("SAUVEGARDE DES RESULTATS")
print("-" * 70)

# Créer le dossier de sortie
output_dir = os.path.join(PROJECT_ROOT, 'results', 'outputs')
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, 'question2_results.txt')

with open(output_path, 'w', encoding='utf-8') as f:
    f.write("=" * 70 + "\n")
    f.write("QUESTION 2: RESULTATS DU CALCUL DU LIKELIHOOD\n")
    f.write("=" * 70 + "\n\n")
    
    f.write("PARAMETRES DU RESEAU:\n")
    f.write("-" * 70 + "\n")
    f.write(f"beta_0  : {beta_0}\n")
    f.write(f"omega_0 : {omega_0}\n")
    f.write(f"beta_1  : {beta_1}\n")
    f.write(f"omega_1 : {omega_1}\n\n")
    
    f.write("DONNEES D'ENTRAINEMENT:\n")
    f.write("-" * 70 + "\n")
    f.write(f"Nombre d'exemples : {len(x_train)}\n")
    f.write(f"Classe 0 : {np.sum(y_train == 0)} exemples\n")
    f.write(f"Classe 1 : {np.sum(y_train == 1)} exemples\n\n")
    
    f.write("RESULTATS:\n")
    f.write("-" * 70 + "\n")
    f.write(f"Likelihood L(theta)     : {likelihood:.15e}\n")
    f.write(f"Log-Likelihood log(L)   : {np.log(likelihood):.15f}\n\n")
    
    f.write("STATISTIQUES DES PROBABILITES PREDITES:\n")
    f.write("-" * 70 + "\n")
    f.write(f"Minimum : {np.min(lambda_pred):.6f}\n")
    f.write(f"Maximum : {np.max(lambda_pred):.6f}\n")
    f.write(f"Moyenne : {np.mean(lambda_pred):.6f}\n\n")
    
    f.write("TABLEAU DES PREDICTIONS:\n")
    f.write("-" * 70 + "\n")
    f.write(f"{'i':<4} | {'x_i':<12} | {'y_i':<6} | {'lambda_i':<12} | {'Contribution':<15}\n")
    f.write("-" * 70 + "\n")
    
    for i in range(len(x_train)):
        f.write(f"{i:<4} | {x_train[i]:<12.6f} | {y_train[i]:<6} | "
                f"{lambda_pred[i]:<12.6f} | {contributions[i]:<15.6e}\n")
    
    f.write("\n" + "=" * 70 + "\n")
    f.write("FIN DU RAPPORT\n")
    f.write("=" * 70 + "\n")

print(f"\n[SAVED] Resultats sauvegardes: {output_path}")

# ============================================================================
# RÉSUMÉ FINAL
# ============================================================================
print("\n" + "=" * 70)
print("RESUME")
print("=" * 70)
print(f"Likelihood L(theta)     : {likelihood:.6e}")
print(f"Log-Likelihood log(L)   : {np.log(likelihood):.6f}")
print(f"Fichier de sortie       : {output_path}")
print("=" * 70)

print("\n>> CONCLUSION:")
print("   Le Likelihood a ete calcule avec succes.")
print("   Cette valeur mesure a quel point le modele explique les donnees.")
print("   Dans la Question 3, nous calculerons la Negative Log-Likelihood.")
print("=" * 70)
