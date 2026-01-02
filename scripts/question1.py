"""
Question 1: Visualisation de f(x) et sigmoid(f(x))
Auteur : Soumia Hariz
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from model import NeuralNetwork

# ========================================
# DONNÉES DU PROJET
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
# PARAMÈTRES DU MODÈLE (du PDF)
# ========================================
beta_0 = [0.3, -1.0, -0.5]
omega_0 = -1.0
beta_1 = 2.6
omega_1 = [-24.0, -8.0, 50.0]

# ========================================
# CRÉER LE MODÈLE
# ========================================
print("="*60)
print("QUESTION 1 : Visualisation du modèle")
print("="*60)

model = NeuralNetwork(beta_0, omega_0, beta_1, omega_1)

# ========================================
# CALCULER f(x) et λ(x) pour x de 0 à 1
# ========================================
x_range = np.linspace(0, 1, 100)

# Calculer f(x)
f_values = model.forward(x_range)

# Calculer λ(x) = sigmoid(f(x))
lambda_values = model.predict_proba(x_range)

print(f"\nCalculs effectués pour {len(x_range)} points")
print(f"Valeur min de f(x): {np.min(f_values):.4f}")
print(f"Valeur max de f(x): {np.max(f_values):.4f}")
print(f"Valeur min de λ(x): {np.min(lambda_values):.4f}")
print(f"Valeur max de λ(x): {np.max(lambda_values):.4f}")

# ========================================
# VISUALISATION
# ========================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Graphique 1: f(x) (avant sigmoid)
axes[0].plot(x_range, f_values, 'b-', linewidth=2)
axes[0].set_xlabel('x', fontsize=12)
axes[0].set_ylabel('f(x)', fontsize=12)
axes[0].set_title('Sortie du réseau : f(x)', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim(0, 1)

# Graphique 2: λ(x) avec les données d'entraînement
axes[1].plot(x_range, lambda_values, 'b-', linewidth=2, label='λ(x) = P(y=1|x)')
axes[1].scatter(x_train, y_train, 
                color='black', 
                s=80, 
                edgecolors='black', 
                facecolors='none', 
                linewidths=2, 
                label='Données d\'entraînement',
                zorder=5)
axes[1].set_xlabel('x', fontsize=12)
axes[1].set_ylabel('Probabilité λ', fontsize=12)
axes[1].set_title('λ(x) = P(y=1|x) et données d\'entraînement', 
                 fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].legend(fontsize=10)
axes[1].set_xlim(0, 1)
axes[1].set_ylim(-0.1, 1.1)

plt.tight_layout()

# Créer le dossier results/figures s'il n'existe pas
output_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'figures')
os.makedirs(output_dir, exist_ok=True)

# Sauvegarder le graphique
output_path = os.path.join(output_dir, 'question1_sigmoid.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✅ Graphique sauvegardé : {output_path}")

# Afficher le graphique
plt.show()

print("\n" + "="*60)
print("✅ Question 1 terminée !")
print("="*60)