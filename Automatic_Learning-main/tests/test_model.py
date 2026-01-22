"""
Tests unitaires pour le modèle
"""
import sys
import os
# Ajouter le dossier parent au path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Maintenant on peut importer
from src.model import NeuralNetwork
from src.activations import relu, sigmoid
import numpy as np


def test_neural_network():
    """Test de la classe NeuralNetwork"""
    # Paramètres du projet
    beta_0 = [0.3, -1.0, -0.5]
    omega_0 = -1.0
    beta_1 = 2.6
    omega_1 = [-24.0, -8.0, 50.0]
    
    model = NeuralNetwork(beta_0, omega_0, beta_1, omega_1)
    
    # Test avec scalaire
    x = 0.5
    prob = model.predict_proba(x)
    assert 0 <= prob <= 1, "La probabilité doit être entre 0 et 1"
    print(f"✅ Test scalaire: λ({x}) = {prob:.4f}")
    
    # Test avec array
    x_array = np.array([0.2, 0.5, 0.8])
    probs = model.predict_proba(x_array)
    assert len(probs) == len(x_array), "Taille incorrecte"
    assert all(0 <= p <= 1 for p in probs), "Toutes les probabilités doivent être entre 0 et 1"
    print(f"✅ Test array: {len(probs)} probabilités calculées")
    
    print("✅ test_neural_network passed")


if __name__ == "__main__":
    print("="*50)
    print("TESTS DU MODÈLE")
    print("="*50)
    test_neural_network()
    print("\n" + "="*50)
    print("✅ TOUS LES TESTS PASSENT !")
    print("="*50)