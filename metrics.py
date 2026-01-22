"""
Fonctions de calcul des métriques: Likelihood et Negative Log-Likelihood.
"""

import numpy as np
from .activations import sigmoid
from .model import forward_pass

def calculate_likelihood(x_train, y_train, beta0, omega0, beta1, omega1):
    """
    Calcule la vraisemblance (Likelihood) pour les données d'entraînement.
    
    Sous l'hypothèse d'indépendance conditionnelle:
    L(θ) = ∏ P(yi | xi, θ)
    
    Avec loi de Bernoulli:
    P(yi | xi) = p̂i^yi * (1 - p̂i)^(1-yi)
    où p̂i = σ(f(xi)) est la probabilité prédite
    
    Args:
        x_train: Données d'entrée (array de N éléments)
        y_train: Labels binaires (array de N éléments: 0 ou 1)
        beta0, omega0, beta1, omega1: Paramètres du réseau
    
    Returns:
        likelihood: Produit des probabilités individuelles
    """
    # Calcul de f(x) pour toutes les entrées
    f_x = forward_pass(x_train, beta0, omega0, beta1, omega1)
    
    # Application de la fonction sigmoïde pour obtenir les probabilités
    # p̂ = σ(f(x)) = P(y=1|x)
    p_pred = sigmoid(f_x)
    
    # Calcul de la probabilité pour chaque observation selon Bernoulli
    # P(yi|xi) = p̂^yi * (1-p̂)^(1-yi)
    probabilities = (p_pred ** y_train) * ((1 - p_pred) ** (1 - y_train))
    
    # Vraisemblance = produit de toutes les probabilités
    likelihood = np.prod(probabilities)
    
    return likelihood


def calculate_negative_log_likelihood(x_train, y_train, beta0, omega0, beta1, omega1):
    """
    Calcule la Negative Log-Likelihood (NLL) - Binary Cross-Entropy.
    
    NLL = -log L(θ) = -∑[yi*log(p̂i) + (1-yi)*log(1-p̂i)]
    
    Cette fonction est équivalente à la Binary Cross-Entropy (BCE).
    On minimise NLL au lieu de maximiser L pour:
    - Éviter les problèmes de sous-flux numérique
    - Transformer le produit en somme (plus stable)
    
    Args:
        x_train: Données d'entrée (array de N éléments)
        y_train: Labels binaires (array de N éléments: 0 ou 1)
        beta0, omega0, beta1, omega1: Paramètres du réseau
    
    Returns:
        nll: Negative Log-Likelihood (scalaire positif)
    """
    # Calcul de f(x) pour toutes les entrées
    f_x = forward_pass(x_train, beta0, omega0, beta1, omega1)
    
    # Application de la fonction sigmoïde
    p_pred = sigmoid(f_x)
    
    # Ajout d'un epsilon pour éviter log(0) qui donne -inf
    epsilon = 1e-15
    p_pred = np.clip(p_pred, epsilon, 1 - epsilon)
    
    # Calcul de la Negative Log-Likelihood (Binary Cross-Entropy)
    # NLL = -∑[yi*log(p̂i) + (1-yi)*log(1-p̂i)]
    nll = -np.sum(
        y_train * np.log(p_pred) + (1 - y_train) * np.log(1 - p_pred)
    )
    
    return nll
