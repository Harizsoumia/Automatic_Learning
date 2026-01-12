"""
Question 2: Calcul du Likelihood
Auteur : Soumia Hariz
"""
import sys
import os
print("SCRIPT :", __file__)
print("CWD :", os.getcwd())
print("sys.path AVANT :", sys.path[:3])

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)
print("sys.path APRÈS :", sys.path[:3])
import numpy as np
from src.model import NeuralNetwork
from src.metrics import calculate_likelihood

# ========================================
# DONNÉES
# ========================================
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

# ========================================
# PARAMÈTRES
# ========================================
beta_0 = [0.3, -1.0, -0.5]
omega_0 = -1.0
beta_1 = 2.6
omega_1 = [-24.0, -8.0, 50.0]

# ========================================
# CALCUL
# ========================================
print("="*60)
print("QUESTION 2 : Calcul du Likelihood")
print("="*60)

# Créer le modèle
model = NeuralNetwork(beta_0, omega_0, beta_1, omega_1)

# Calculer les probabilités λ prédites
lambda_pred = model.predict_proba(x_train)

# Calculer le likelihood
likelihood = calculate_likelihood(y_train, lambda_pred, verbose=True)

# Sauvegarder les résultats
output_path = os.path.join('..', 'results', 'outputs', 'question2_results.txt')
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, 'w', encoding='utf-8') as f:
    f.write("QUESTION 2 : RÉSULTATS\n")
    f.write("="*60 + "\n\n")
    f.write(f"Nombre de points d'entraînement : {len(x_train)}\n")
    f.write(f"Likelihood total : {likelihood:.15f}\n")
    f.write(f"Log-Likelihood : {np.log(likelihood):.15f}\n")

print(f"\n✅ Résultats sauvegardés : {output_path}")
print("\n" + "="*60)
print("Question 2 terminée !")
print("="*60)