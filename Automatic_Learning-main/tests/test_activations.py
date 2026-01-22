"""
Tests unitaires pour les fonctions d'activation
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from src.activations import relu, sigmoid


def test_relu():
    """Test de la fonction ReLU"""
    assert relu(-5) == 0, "ReLU(-5) devrait être 0"
    assert relu(0) == 0, "ReLU(0) devrait être 0"
    assert relu(5) == 5, "ReLU(5) devrait être 5"
    
    # Test avec array
    arr = np.array([-2, -1, 0, 1, 2])
    expected = np.array([0, 0, 0, 1, 2])
    assert np.allclose(relu(arr), expected), "ReLU array failed"
    
    print("✅ test_relu passed")


def test_sigmoid():
    """Test de la fonction Sigmoid"""
    assert np.isclose(sigmoid(0), 0.5), "sigmoid(0) devrait être 0.5"
    assert sigmoid(100) > 0.99, "sigmoid(100) devrait être proche de 1"
    assert sigmoid(-100) < 0.01, "sigmoid(-100) devrait être proche de 0"
    
    print("✅ test_sigmoid passed")


if __name__ == "__main__":
    test_relu()
    test_sigmoid()
    print("\n" + "="*50)
    print("✅ TOUS LES TESTS PASSENT !")
    print("="*50)