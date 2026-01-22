"""
Fonctions d'activation pour le réseau de neurones
Auteur : Soumia Hariz

Ce module contient les fonctions d'activation utilisées dans le réseau :
- ReLU : pour la couche cachée (activation non-linéaire)
- Sigmoid : pour la couche de sortie (conversion en probabilité)
"""

import numpy as np


def relu(z):
    """
    Fonction d'activation ReLU (Rectified Linear Unit).
    
    Formule : ReLU(z) = max(0, z)
    
    Propriétés :
    - Active uniquement les valeurs positives
    - Dérivée simple : 1 si z > 0, sinon 0
    - Évite le problème de gradient vanishing
    - Utilisée dans la couche cachée du réseau
    
    Args:
        z: Entrée scalaire ou array numpy (pré-activation)
    
    Returns:
        Array numpy ou scalaire : valeur activée (≥ 0)
    
    Exemples:
        >>> relu(-2)
        0
        >>> relu(3.5)
        3.5
        >>> relu(np.array([-1, 0, 2]))
        array([0, 0, 2])
    """
    return np.maximum(0, z)


def sigmoid(z):
    """
    Fonction d'activation Sigmoid (fonction logistique).
    
    Formule : σ(z) = 1 / (1 + exp(-z))
    
    Propriétés :
    - Sortie dans l'intervalle (0, 1)
    - Interprétable comme une probabilité
    - σ(0) = 0.5
    - Asymptotes : σ(-∞) → 0, σ(+∞) → 1
    - Utilisée pour la classification binaire
    
    Args:
        z: Entrée scalaire ou array numpy (logit)
    
    Returns:
        Array numpy ou scalaire : probabilité entre 0 et 1
    
    Exemples:
        >>> sigmoid(0)
        0.5
        >>> sigmoid(5)
        0.9933...
        >>> sigmoid(-5)
        0.0066...
    
    Note:
        Pour éviter les overflow avec exp(-z) quand z est très grand,
        numpy gère automatiquement ces cas.
    """
    # Protection contre les overflow (optionnel, numpy le gère déjà bien)
    # z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


# ============================================================================
# TESTS UNITAIRES
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("TESTS DES FONCTIONS D'ACTIVATION")
    print("=" * 60)
    
    # Test 1 : ReLU avec valeurs scalaires
    print("\n📊 Test 1: ReLU (valeurs scalaires)")
    print("-" * 60)
    test_values_relu = [-5, -2, 0, 2, 5]
    print(f"{'Entrée':<10} | {'ReLU(z)':<10}")
    print("-" * 25)
    for val in test_values_relu:
        result = relu(val)
        print(f"{val:<10} | {result:<10}")
    
    # Test 2 : ReLU avec array numpy
    print("\n📊 Test 2: ReLU (array numpy)")
    print("-" * 60)
    z_array = np.array([-3, -1, 0, 1, 3])
    result_array = relu(z_array)
    print(f"Entrée  : {z_array}")
    print(f"ReLU(z) : {result_array}")
    
    # Test 3 : Sigmoid avec valeurs scalaires
    print("\n📊 Test 3: Sigmoid (valeurs scalaires)")
    print("-" * 60)
    test_values_sigmoid = [-10, -5, -1, 0, 1, 5, 10]
    print(f"{'Entrée':<10} | {'σ(z)':<15}")
    print("-" * 30)
    for val in test_values_sigmoid:
        result = sigmoid(val)
        print(f"{val:<10} | {result:<15.10f}")
    
    # Test 4 : Sigmoid avec array numpy
    print("\n📊 Test 4: Sigmoid (array numpy)")
    print("-" * 60)
    z_sig = np.array([-5, -2, 0, 2, 5])
    result_sig = sigmoid(z_sig)
    print(f"Entrée  : {z_sig}")
    print(f"σ(z)    : {result_sig}")
    
    # Test 5 : Propriétés de Sigmoid
    print("\n📊 Test 5: Vérification des propriétés")
    print("-" * 60)
    sig_0 = sigmoid(0)
    sig_pos = sigmoid(100)  # Très grand positif
    sig_neg = sigmoid(-100)  # Très grand négatif
    
    print(f"σ(0) = {sig_0} (devrait être 0.5)")
    print(f"σ(100) ≈ {sig_pos:.10f} (devrait être proche de 1)")
    print(f"σ(-100) ≈ {sig_neg:.10f} (devrait être proche de 0)")
    
    # Vérification : σ(z) + σ(-z) = 1
    z_test = 3.7
    sum_test = sigmoid(z_test) + sigmoid(-z_test)
    print(f"\nPropriété symétrique:")
    print(f"σ({z_test}) + σ({-z_test}) = {sum_test:.10f}")
    print(f"Devrait être égal à 1.0 : {np.isclose(sum_test, 1.0)}")
    
    # Test 6 : Performance avec grandes matrices
    print("\n📊 Test 6: Performance avec grande matrice")
    print("-" * 60)
    large_array = np.random.randn(1000, 100)
    
    import time
    start = time.time()
    relu_result = relu(large_array)
    relu_time = time.time() - start
    
    start = time.time()
    sig_result = sigmoid(large_array)
    sig_time = time.time() - start
    
    print(f"Matrice de taille: {large_array.shape}")
    print(f"Temps ReLU    : {relu_time*1000:.4f} ms")
    print(f"Temps Sigmoid : {sig_time*1000:.4f} ms")
    
    print("\n" + "=" * 60)
    print("✅ TOUS LES TESTS SONT TERMINÉS")
    print("=" * 60)
    
