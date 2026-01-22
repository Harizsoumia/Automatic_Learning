"""
Fonctions de visualisation pour les résultats du réseau de neurones.
Auteur: Soumia Hariz
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from .activations import sigmoid
from .model import NeuralNetwork


def plot_sigmoid_with_data(x_train, y_train, beta0, omega0, beta1, omega1, save_path=None):
    """
    Trace la courbe sigmoid sigma(f(x)) et les points d'entraînement.
    
    Args:
        x_train: Données d'entrée d'entraînement
        y_train: Labels d'entraînement
        beta0, omega0, beta1, omega1: Paramètres du réseau
        save_path: Chemin pour sauvegarder l'image (optionnel)
    """
    # Créer le réseau de neurones
    nn = NeuralNetwork(beta0, omega0, beta1, omega1)
    
    # Générer des points pour tracer la courbe continue
    x_curve = np.linspace(0, 1, 100)
    p_curve = nn.predict_proba(x_curve)
    
    # Création du graphique
    plt.figure(figsize=(10, 6))
    plt.plot(x_curve, p_curve, 'b-', linewidth=2, label='P(y=1|x) = sigma(f(x))')
    plt.scatter(x_train, y_train, c='black', s=100, edgecolors='black', 
                linewidths=1.5, label="Donnees d'entrainement", zorder=5)
    
    plt.xlabel('x', fontsize=12)
    plt.ylabel('Probabilite P(y=1|x)', fontsize=12)
    plt.title("Sortie du reseau apres sigmoid et donnees d'entrainement", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([-0.05, 1.05])
    
    # Sauvegarder si un chemin est fourni
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[SAVED] Graphique sauvegarde: {save_path}")
    
    plt.show()


def plot_likelihood_vs_beta1(x_train, y_train, beta0, omega0, omega1, 
                              beta1_range=None, save_dir='results'):
    """
    Trace DEUX graphiques séparés de Likelihood et NLL en fonction de beta1.
    Sauvegarde automatiquement les images dans le dossier 'results'.
    
    Args:
        x_train: Données d'entrée d'entraînement
        y_train: Labels d'entraînement
        beta0, omega0, omega1: Paramètres fixes du réseau
        beta1_range: Tuple (min, max, nb_points) pour beta1
        save_dir: Dossier où sauvegarder les images
    
    Returns:
        (beta1_max_L, beta1_min_NLL): Valeurs optimales de beta1
    """
    from .metrics import calculate_likelihood, calculate_negative_log_likelihood
    
    # Créer le dossier de sauvegarde s'il n'existe pas
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"[INFO] Dossier cree: {save_dir}/")
    
    # Définir la plage de beta1 à explorer
    if beta1_range is None:
        beta1_values = np.linspace(-30, 30, 500)
    else:
        beta1_values = np.linspace(beta1_range[0], beta1_range[1], beta1_range[2])
    
    # Calculer Likelihood et NLL pour chaque valeur de beta1
    likelihoods = []
    nlls = []
    
    print("\n[INFO] Calcul en cours pour {} valeurs de beta_1...".format(len(beta1_values)))
    
    for b1 in beta1_values:
        # Calcul des métriques
        L = calculate_likelihood(x_train, y_train, beta0, omega0, b1, omega1)
        NLL = calculate_negative_log_likelihood(x_train, y_train, beta0, omega0, b1, omega1)
        
        likelihoods.append(L)
        nlls.append(NLL)
    
    # Convertir en arrays numpy
    likelihoods = np.array(likelihoods)
    nlls = np.array(nlls)
    
    # Trouver les valeurs optimales de beta1
    max_L_idx = np.argmax(likelihoods)
    min_NLL_idx = np.argmin(nlls)
    
    beta1_max_L = beta1_values[max_L_idx]
    beta1_min_NLL = beta1_values[min_NLL_idx]
    
    # Affichage des résultats
    print("\n" + "=" * 60)
    print("RESULTATS DE L'OPTIMISATION")
    print("=" * 60)
    print(f"Beta1 qui maximise le Likelihood: {beta1_max_L:.4f}")
    print(f"  -> Likelihood maximum: {likelihoods[max_L_idx]:.6e}")
    print(f"\nBeta1 qui minimise le NLL: {beta1_min_NLL:.4f}")
    print(f"  -> NLL minimum: {nlls[min_NLL_idx]:.6f}")
    print("=" * 60)
    
    # ========================================================================
    # GRAPHIQUE 1: LIKELIHOOD
    # ========================================================================
    plt.figure(figsize=(12, 6))
    
    plt.plot(beta1_values, likelihoods, 'b-', linewidth=2.5, label='Likelihood L(theta)')
    plt.axvline(beta1_max_L, color='red', linestyle='--', linewidth=2, 
                label=f'Maximum a beta_1 = {beta1_max_L:.4f}')
    plt.scatter([beta1_max_L], [likelihoods[max_L_idx]], color='red', 
                s=200, zorder=5, marker='o', edgecolors='darkred', linewidths=2)
    
    # Annotation du point maximum
    plt.annotate(f'Max = {likelihoods[max_L_idx]:.6e}\nbeta_1 = {beta1_max_L:.4f}',
                xy=(beta1_max_L, likelihoods[max_L_idx]),
                xytext=(beta1_max_L + 5, likelihoods[max_L_idx] * 0.8),
                fontsize=11,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    
    plt.xlabel('beta_1 (biais de sortie)', fontsize=13, fontweight='bold')
    plt.ylabel('Likelihood L(theta)', fontsize=13, fontweight='bold')
    plt.title('Likelihood en fonction de beta_1', fontsize=15, fontweight='bold')
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Sauvegarder le graphique 1
    save_path_L = os.path.join(save_dir, 'likelihood_vs_beta1.png')
    plt.savefig(save_path_L, dpi=300, bbox_inches='tight')
    print(f"\n[SAVED] Graphique Likelihood sauvegarde: {save_path_L}")
    
    plt.show()
    
    # ========================================================================
    # GRAPHIQUE 2: NEGATIVE LOG-LIKELIHOOD
    # ========================================================================
    plt.figure(figsize=(12, 6))
    
    plt.plot(beta1_values, nlls, 'g-', linewidth=2.5, label='Negative Log-Likelihood (NLL)')
    plt.axvline(beta1_min_NLL, color='red', linestyle='--', linewidth=2, 
                label=f'Minimum a beta_1 = {beta1_min_NLL:.4f}')
    plt.scatter([beta1_min_NLL], [nlls[min_NLL_idx]], color='red', 
                s=200, zorder=5, marker='o', edgecolors='darkred', linewidths=2)
    
    # Annotation du point minimum
    plt.annotate(f'Min = {nlls[min_NLL_idx]:.4f}\nbeta_1 = {beta1_min_NLL:.4f}',
                xy=(beta1_min_NLL, nlls[min_NLL_idx]),
                xytext=(beta1_min_NLL + 5, nlls[min_NLL_idx] + 50),
                fontsize=11,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    
    plt.xlabel('beta_1 (biais de sortie)', fontsize=13, fontweight='bold')
    plt.ylabel('Negative Log-Likelihood (NLL)', fontsize=13, fontweight='bold')
    plt.title('Negative Log-Likelihood en fonction de beta_1', fontsize=15, fontweight='bold')
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Sauvegarder le graphique 2
    save_path_NLL = os.path.join(save_dir, 'nll_vs_beta1.png')
    plt.savefig(save_path_NLL, dpi=300, bbox_inches='tight')
    print(f"[SAVED] Graphique NLL sauvegarde: {save_path_NLL}")
    
    plt.show()
    
    print("\n[INFO] Les deux graphiques ont ete sauvegardes avec succes!")
    
    return beta1_max_L, beta1_min_NLL


# ============================================================================
# TESTS
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("TEST DES FONCTIONS DE VISUALISATION")
    print("=" * 70)
    
    # Données de test
    x_train = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    y_train = np.array([0, 0, 1, 1, 1])
    
    beta0 = np.array([0.3, -1.0, -0.5])
    omega0 = np.array([-1.0, 1.8, 0.65])
    beta1 = 2.6
    omega1 = np.array([-24.0, -8.0, 50.0])
    
    print("\nTest 1: Visualisation sigmoid avec donnees")
    print("-" * 70)
    plot_sigmoid_with_data(x_train, y_train, beta0, omega0, beta1, omega1, 
                          save_path='results/test_sigmoid.png')
    
    print("\nTest 2: Visualisation Likelihood vs beta1")
    print("-" * 70)
    beta1_opt_L, beta1_opt_NLL = plot_likelihood_vs_beta1(
        x_train, y_train, beta0, omega0, omega1,
        beta1_range=(-10, 10, 100)
    )
    
    print("\n" + "=" * 70)
    print("TESTS TERMINES")
    print("=" * 70)
    
