"""
Question 1: Visualisation de f(x) et sigmoid(f(x))

Ce script:
1. Calcule f(x) pour x dans [0, 1]
2. Calcule lambda(x) = sigma(f(x)) = P(y=1|x)
3. Trace les deux courbes
4. Superpose les données d'entraînement sur lambda(x)
5. Sauvegarde les graphiques

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
import matplotlib.pyplot as plt
from src.model import NeuralNetwork

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
# CRÉATION DU MODÈLE
# ============================================================================
print("=" * 70)
print("QUESTION 1: VISUALISATION DE f(x) ET sigma(f(x))")
print("=" * 70)

print("\nParametres du reseau:")
print(f"  beta_0  : {beta_0}")
print(f"  omega_0 : {omega_0}")
print(f"  beta_1  : {beta_1}")
print(f"  omega_1 : {omega_1}")

# Créer le réseau de neurones
model = NeuralNetwork(beta_0, omega_0, beta_1, omega_1)
print("\n[OK] Modele cree avec succes")

# ============================================================================
# CALCUL DE f(x) ET lambda(x) POUR x dans [0, 1]
# ============================================================================
print("\n" + "-" * 70)
print("CALCUL DES SORTIES DU RESEAU")
print("-" * 70)

# Générer 100 points entre 0 et 1
x_range = np.linspace(0, 1, 100)

# Calculer f(x) : sortie avant sigmoid
f_values = model.forward(x_range)

# Calculer lambda(x) = sigma(f(x)) : probabilité P(y=1|x)
lambda_values = model.predict_proba(x_range)

print(f"Nombre de points calcules : {len(x_range)}")
print(f"\nStatistiques de f(x):")
print(f"  Minimum : {np.min(f_values):.4f}")
print(f"  Maximum : {np.max(f_values):.4f}")
print(f"  Moyenne : {np.mean(f_values):.4f}")

print(f"\nStatistiques de lambda(x) = P(y=1|x):")
print(f"  Minimum : {np.min(lambda_values):.4f}")
print(f"  Maximum : {np.max(lambda_values):.4f}")
print(f"  Moyenne : {np.mean(lambda_values):.4f}")

# ============================================================================
# VISUALISATION: DEUX GRAPHIQUES SÉPARÉS
# ============================================================================
print("\n" + "-" * 70)
print("CREATION DES GRAPHIQUES")
print("-" * 70)

# Créer le dossier de sauvegarde
output_dir = os.path.join(PROJECT_ROOT, 'results', 'figures')
os.makedirs(output_dir, exist_ok=True)

# ------------------------------------------------------------------------
# GRAPHIQUE 1: f(x) - Sortie avant sigmoid
# ------------------------------------------------------------------------
plt.figure(figsize=(10, 6))

plt.plot(x_range, f_values, 'b-', linewidth=2.5, label='f(x)')
plt.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

plt.xlabel('x', fontsize=13, fontweight='bold')
plt.ylabel('f(x)', fontsize=13, fontweight='bold')
plt.title('Sortie du reseau avant sigmoid: f(x)', fontsize=15, fontweight='bold')
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(fontsize=11)
plt.xlim(0, 1)

# Sauvegarder
output_path_fx = os.path.join(output_dir, 'question1_fx.png')
plt.savefig(output_path_fx, dpi=300, bbox_inches='tight')
print(f"\n[SAVED] Graphique f(x) sauvegarde: {output_path_fx}")

plt.show()

# ------------------------------------------------------------------------
# GRAPHIQUE 2: lambda(x) avec données d'entraînement
# ------------------------------------------------------------------------
plt.figure(figsize=(10, 6))

# Courbe de probabilité
plt.plot(x_range, lambda_values, 'b-', linewidth=2.5, 
         label='lambda(x) = P(y=1|x)')

# Points d'entraînement (cercles noirs)
plt.scatter(x_train, y_train, 
            color='black', 
            s=100, 
            edgecolors='black', 
            facecolors='none', 
            linewidths=2, 
            label="Donnees d'entrainement",
            zorder=5)

# Ligne à y=0.5 (seuil de décision)
plt.axhline(y=0.5, color='red', linestyle='--', linewidth=1, 
            alpha=0.5, label='Seuil de decision (0.5)')

plt.xlabel('x', fontsize=13, fontweight='bold')
plt.ylabel('Probabilite lambda(x)', fontsize=13, fontweight='bold')
plt.title("Probabilite P(y=1|x) et donnees d'entrainement", 
          fontsize=15, fontweight='bold')
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(fontsize=11, loc='best')
plt.xlim(0, 1)
plt.ylim(-0.05, 1.05)

# Sauvegarder
output_path_lambda = os.path.join(output_dir, 'question1_sigmoid.png')
plt.savefig(output_path_lambda, dpi=300, bbox_inches='tight')
print(f"[SAVED] Graphique lambda(x) sauvegarde: {output_path_lambda}")

plt.show()

# ============================================================================
# ANALYSE DES PRÉDICTIONS
# ============================================================================
print("\n" + "-" * 70)
print("ANALYSE DES PREDICTIONS SUR LES DONNEES D'ENTRAINEMENT")
print("-" * 70)

# Calculer les probabilités pour les données d'entraînement
probas_train = model.predict_proba(x_train)
predictions_train = model.predict(x_train, threshold=0.5)

# Calculer l'accuracy
accuracy = np.mean(predictions_train == y_train) * 100

print(f"\nAccuracy sur les donnees d'entrainement: {accuracy:.2f}%")

# Afficher quelques exemples
print("\nExemples de predictions:")
print(f"{'x':<12} | {'y reel':<8} | {'P(y=1|x)':<12} | {'Prediction':<12} | {'Correct':<8}")
print("-" * 70)

for i in range(min(10, len(x_train))):
    correct = "OUI" if predictions_train[i] == y_train[i] else "NON"
    print(f"{x_train[i]:<12.6f} | {y_train[i]:<8} | {probas_train[i]:<12.6f} | "
          f"{predictions_train[i]:<12} | {correct:<8}")

# ============================================================================
# RÉSUMÉ FINAL
# ============================================================================
print("\n" + "=" * 70)
print("RESUME")
print("=" * 70)
print(f"Graphiques sauvegardes dans: {output_dir}/")
print(f"  - question1_fx.png       : Sortie f(x) avant sigmoid")
print(f"  - question1_sigmoid.png  : Probabilite lambda(x) avec donnees")
print(f"\nAccuracy du modele: {accuracy:.2f}%")
print("=" * 70)

print("\n>> CONCLUSION:")
print("   Le reseau de neurones transforme les entrees x en probabilites.")
print("   La fonction sigmoid convertit f(x) en probabilites entre 0 et 1.")
print("   Les points noirs montrent les vraies valeurs de y.")
print("=" * 70)