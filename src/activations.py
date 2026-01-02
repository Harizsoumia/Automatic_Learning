"""
Fonctions d'activation pour le réseau de neurones
Auteur : Soumia Hariz
"""
import numpy as np

def relu(z):
    """
    ReLU activation: max(0, z)
    
    Args:
        z: scalaire ou array numpy
    Returns:
        max(0, z)
    """
    return np.maximum(0, z)


def sigmoid(z):
    """
    Sigmoid activation: 1 / (1 + exp(-z))
    
    Args:
        z: scalaire ou array numpy
    Returns:
        valeur entre 0 et 1
    """
    return 1 / (1 + np.exp(-z))


# Tests rapides
if __name__ == "__main__":
    print("Test ReLU:")
    print(f"relu(-2) = {relu(-2)}")  # Devrait donner 0
    print(f"relu(3) = {relu(3)}")    # Devrait donner 3
    
    print("\nTest Sigmoid:")
    print(f"sigmoid(0) = {sigmoid(0)}")      # Devrait donner 0.5
    print(f"sigmoid(5) = {sigmoid(5):.4f}")  # Devrait donner ~0.9933
    print(f"sigmoid(-5) = {sigmoid(-5):.4f}") # Devrait donner ~0.0067