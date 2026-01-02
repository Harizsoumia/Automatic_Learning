"""
Calcul du Likelihood et autres métriques
Auteur : Soumia Hariz - Question 2
"""
import numpy as np


def bernoulli_probability(y, lambd):
    """
    Calcule P(y | λ) selon la loi de Bernoulli
    
    P(y | λ) = λ^y × (1-λ)^(1-y)
    
    Args:
        y: valeur observée (0 ou 1)
        lambd: probabilité prédite P(y=1|x) = λ
    Returns:
        probabilité selon Bernoulli
    """
    return (lambd ** y) * ((1 - lambd) ** (1 - y))


def calculate_likelihood(y_true, lambda_pred, verbose=True):
    """
    Calcule le Likelihood total
    
    L(Θ) = ∏ P(yi | λi)
    
    Args:
        y_true: array des vraies valeurs (0 ou 1)
        lambda_pred: array des probabilités prédites λ
        verbose: afficher les détails
    Returns:
        likelihood total (produit des probabilités)
    """
    likelihood = 1.0
    
    if verbose:
        print("\n" + "="*60)
        print("CALCUL DU LIKELIHOOD")
        print("="*60)
    
    for i in range(len(y_true)):
        prob = bernoulli_probability(y_true[i], lambda_pred[i])
        likelihood *= prob
        
        if verbose:
            print(f"Point {i+1:2d}: y={y_true[i]}, λ={lambda_pred[i]:.4f} → P={prob:.6f}")
    
    if verbose:
        print("="*60)
        print(f"LIKELIHOOD TOTAL = {likelihood:.10f}")
        print("="*60)
    
    return likelihood


# Tests
if __name__ == "__main__":
    # Test simple
    y_test = np.array([1, 0, 1])
    lambda_test = np.array([0.9, 0.1, 0.8])
    
    L = calculate_likelihood(y_test, lambda_test, verbose=True)
    
    print(f"\n✅ Likelihood calculé : {L:.6f}")