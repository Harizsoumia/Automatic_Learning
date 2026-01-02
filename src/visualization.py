"""
Fonctions de visualisation
Auteur : Soumia Hariz
"""
import matplotlib.pyplot as plt
import numpy as np


def plot_sigmoid_with_data(x_range, lambda_values, x_train, y_train, save_path=None):
    """
    Trace λ(x) = sigmoid(f(x)) avec les données d'entraînement
    
    Args:
        x_range: array des x pour la courbe
        lambda_values: array des λ calculés
        x_train: données d'entraînement x
        y_train: données d'entraînement y
        save_path: chemin pour sauvegarder (optionnel)
    """
    plt.figure(figsize=(10, 6))
    
    # Courbe du modèle
    plt.plot(x_range, lambda_values, 'b-', linewidth=2, label='λ(x) = P(y=1|x)')
    
    # Points d'entraînement
    plt.scatter(x_train, y_train, 
                color='black', 
                s=80, 
                edgecolors='black', 
                facecolors='none', 
                linewidths=2, 
                label='Données d\'entraînement',
                zorder=5)
    
    plt.xlabel('x', fontsize=12)
    plt.ylabel('Probabilité λ', fontsize=12)
    plt.title('λ(x) = P(y=1|x) et données d\'entraînement', 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.xlim(0, 1)
    plt.ylim(-0.1, 1.1)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Graphique sauvegardé : {save_path}")
    
    plt.show()


def plot_optimization(beta_1_values, likelihoods, nlls, save_path=None):
    """
    Trace Likelihood et NLL en fonction de β₁
    
    Args:
        beta_1_values: array des valeurs de β₁ testées
        likelihoods: array des Likelihood correspondants
        nlls: array des NLL correspondants
        save_path: chemin pour sauvegarder (optionnel)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Trouver les optimaux
    best_idx_L = np.argmax(likelihoods)
    best_beta_1_L = beta_1_values[best_idx_L]
    
    best_idx_NLL = np.argmin(nlls)
    best_beta_1_NLL = beta_1_values[best_idx_NLL]
    
    # Graphique 1: Likelihood
    axes[0].plot(beta_1_values, likelihoods, 'g-', linewidth=2)
    axes[0].axvline(best_beta_1_L, color='red', linestyle='--', linewidth=2,
                    label=f'Max à β₁={best_beta_1_L:.2f}')
    axes[0].set_xlabel('β₁', fontsize=12)
    axes[0].set_ylabel('Likelihood', fontsize=12)
    axes[0].set_title('Likelihood (À MAXIMISER)', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=10)
    
    # Graphique 2: NLL
    axes[1].plot(beta_1_values, nlls, 'r-', linewidth=2)
    axes[1].axvline(best_beta_1_NLL, color='blue', linestyle='--', linewidth=2,
                    label=f'Min à β₁={best_beta_1_NLL:.2f}')
    axes[1].set_xlabel('β₁', fontsize=12)
    axes[1].set_ylabel('NLL', fontsize=12)
    axes[1].set_title('Negative Log-Likelihood (À MINIMISER)', 
                     fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Graphique sauvegardé : {save_path}")
    
    plt.show()
    
    return best_beta_1_L, best_beta_1_NLL