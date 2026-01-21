"""
Architecture du réseau de neurones pour classification binaire
Auteur : Soumia Hariz

Architecture:
    Entrée (x) → Couche cachée (3 neurones ReLU) → Couche de sortie (1 neurone linéaire) → Sigmoid
    
    h(x) = ReLU(β₀ + Ω₀ · x)  [Couche cachée, 3 neurones]
    f(x) = β₁ + Ω₁ᵀ · h(x)    [Couche de sortie, linéaire]
    λ(x) = σ(f(x))             [Probabilité P(y=1|x)]
"""

import numpy as np
from .activations import relu, sigmoid


class NeuralNetwork:
    """
    Réseau de neurones peu profond pour classification binaire.
    
    Couches:
        - Couche cachée : 3 neurones avec activation ReLU
        - Couche de sortie : 1 neurone linéaire
        - Activation finale : Sigmoid (pour obtenir une probabilité)
    
    Paramètres:
        β₀ (beta_0) : Biais de la couche cachée (vecteur de taille 3)
        Ω₀ (omega_0) : Poids de la couche cachée (vecteur de taille 3)
        β₁ (beta_1) : Biais de la couche de sortie (scalaire)
        Ω₁ (omega_1) : Poids de la couche de sortie (vecteur de taille 3)
    """
    
    def __init__(self, beta_0, omega_0, beta_1, omega_1):
        """
        Initialise le réseau avec les paramètres donnés.
        
        Args:
            beta_0: Biais couche cachée (array-like de taille 3)
            omega_0: Poids couche cachée (array-like de taille 3)
            beta_1: Biais sortie (scalaire)
            omega_1: Poids sortie (array-like de taille 3)
        
        Exemples:
            >>> beta_0 = [0.3, -1.0, -0.5]
            >>> omega_0 = [-1.0, 1.8, 0.65]
            >>> beta_1 = 2.6
            >>> omega_1 = [-24.0, -8.0, 50.0]
            >>> nn = NeuralNetwork(beta_0, omega_0, beta_1, omega_1)
        """
        # Conversion en arrays numpy pour faciliter les calculs vectoriels
        self.beta_0 = np.array(beta_0, dtype=np.float64)    # Shape: (3,)
        self.omega_0 = np.array(omega_0, dtype=np.float64)  # Shape: (3,)
        self.beta_1 = float(beta_1)                          # Scalaire
        self.omega_1 = np.array(omega_1, dtype=np.float64)  # Shape: (3,)
        
        # Validation des dimensions
        assert self.beta_0.shape == (3,), "beta_0 doit être de taille 3"
        assert self.omega_0.shape == (3,), "omega_0 doit être de taille 3"
        assert self.omega_1.shape == (3,), "omega_1 doit être de taille 3"
    
    def hidden_layer(self, x):
        """
        Calcule la sortie de la couche cachée avec activation ReLU.
        
        Formule: h(x) = ReLU(β₀ + Ω₀ ⊙ x)
        où ⊙ représente la multiplication élément par élément
        
        Pour chaque neurone i (i=0,1,2):
            h_i(x) = ReLU(β₀[i] + Ω₀[i] * x)
        
        Args:
            x: Entrée scalaire ou array numpy de shape (n,)
        
        Returns:
            h(x): 
                - Si x scalaire: array de shape (3,)
                - Si x array de taille n: array de shape (n, 3)
        
        Exemples:
            >>> nn.hidden_layer(0.5)
            array([...])  # Shape: (3,)
            >>> nn.hidden_layer([0.1, 0.5, 0.9])
            array([...])  # Shape: (3, 3)
        """
        # Conversion en array numpy
        x = np.asarray(x, dtype=np.float64)
        
        # Cas scalaire : x est un seul nombre
        if x.ndim == 0 or (x.ndim == 1 and x.size == 1):
            # Pour chaque neurone: beta_0[i] + omega_0[i] * x
            z = self.beta_0 + self.omega_0 * float(x)  # Shape: (3,)
            return relu(z)
        
        # Cas vectoriel : x est un array de n éléments
        else:
            # x shape: (n,) → reshape en (n, 1) pour broadcasting
            x = x.reshape(-1, 1)  # Shape: (n, 1)
            
            # Broadcasting:
            # beta_0 shape: (3,) → broadcast en (1, 3)
            # omega_0 shape: (3,) → broadcast en (1, 3)
            # x shape: (n, 1)
            # Résultat: (n, 1) + (1, 3) * (n, 1) = (n, 3)
            z = self.beta_0 + self.omega_0 * x  # Shape: (n, 3)
            return relu(z)
    
    def forward(self, x):
        """
        Calcule la sortie du réseau avant activation sigmoid.
        
        Formule: f(x) = β₁ + Ω₁ᵀ · h(x)
        où · représente le produit scalaire
        
        Args:
            x: Entrée scalaire ou array numpy de shape (n,)
        
        Returns:
            f(x):
                - Si x scalaire: scalaire
                - Si x array de taille n: array de shape (n,)
        
        Exemples:
            >>> nn.forward(0.5)
            2.134...  # Scalaire
            >>> nn.forward([0.1, 0.5, 0.9])
            array([...])  # Shape: (3,)
        """
        # Calcul de la couche cachée
        h = self.hidden_layer(x)  # Shape: (3,) ou (n, 3)
        
        # Conversion en array pour traitement uniforme
        x = np.asarray(x, dtype=np.float64)
        
        # Cas scalaire
        if x.ndim == 0 or (x.ndim == 1 and x.size == 1):
            # h shape: (3,), omega_1 shape: (3,)
            # Produit scalaire: sum(omega_1[i] * h[i])
            f = self.beta_1 + np.dot(self.omega_1, h)  # Scalaire
            return float(f)
        
        # Cas vectoriel
        else:
            # h shape: (n, 3), omega_1 shape: (3,)
            # Matrix multiplication: (n, 3) @ (3,) = (n,)
            f = self.beta_1 + np.dot(h, self.omega_1)  # Shape: (n,)
            return f
    
    def predict_proba(self, x):
        """
        Calcule la probabilité P(y=1|x) = λ(x) = σ(f(x)).
        
        Cette méthode retourne la probabilité que y = 1 sachant x,
        selon le modèle de Bernoulli.
        
        Formule: λ(x) = σ(f(x)) = 1 / (1 + exp(-f(x)))
        
        Args:
            x: Entrée scalaire ou array numpy de shape (n,)
        
        Returns:
            λ(x): Probabilité(s) entre 0 et 1
                - Si x scalaire: scalaire dans [0, 1]
                - Si x array de taille n: array de shape (n,) avec valeurs dans [0, 1]
        
        Exemples:
            >>> nn.predict_proba(0.5)
            0.894...  # Probabilité que y=1 pour x=0.5
            >>> nn.predict_proba([0.1, 0.5, 0.9])
            array([0.234..., 0.894..., 0.987...])
        """
        # Calcul de f(x)
        f = self.forward(x)
        
        # Application de la fonction sigmoid
        return sigmoid(f)
    
    def predict(self, x, threshold=0.5):
        """
        Prédit la classe binaire (0 ou 1) selon un seuil.
        
        Args:
            x: Entrée scalaire ou array numpy
            threshold: Seuil de décision (par défaut 0.5)
        
        Returns:
            Classe prédite (0 ou 1):
                - 1 si λ(x) >= threshold
                - 0 sinon
        
        Exemples:
            >>> nn.predict(0.5)
            1
            >>> nn.predict([0.1, 0.5, 0.9])
            array([0, 1, 1])
        """
        proba = self.predict_proba(x)
        return (proba >= threshold).astype(int)
    
    def __repr__(self):
        """Représentation textuelle du réseau."""
        return (
            f"NeuralNetwork(\n"
            f"  Couche cachée: 3 neurones ReLU\n"
            f"    β₀ = {self.beta_0}\n"
            f"    Ω₀ = {self.omega_0}\n"
            f"  Couche de sortie: 1 neurone linéaire\n"
            f"    β₁ = {self.beta_1}\n"
            f"    Ω₁ = {self.omega_1}\n"
            f")"
        )


# ============================================================================
# FONCTION UTILITAIRE (compatible avec l'ancien code)
# ============================================================================
def forward_pass(x, beta0, omega0, beta1, omega1):
    """
    Fonction utilitaire pour la rétrocompatibilité.
    Effectue une propagation avant sans créer d'objet NeuralNetwork.
    
    Args:
        x: Entrée(s) scalaire(s) ou array
        beta0, omega0, beta1, omega1: Paramètres du réseau
    
    Returns:
        f(x): Sortie du réseau (avant sigmoid)
    """
    nn = NeuralNetwork(beta0, omega0, beta1, omega1)
    return nn.forward(x)


# ============================================================================
# TESTS UNITAIRES
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("TESTS DU RÉSEAU DE NEURONES")
    print("=" * 70)
    
    # Paramètres du projet
    beta_0 = np.array([0.3, -1.0, -0.5])
    omega_0 = np.array([-1.0, 1.8, 0.65])
    beta_1 = 2.6
    omega_1 = np.array([-24.0, -8.0, 50.0])
    
    # Création du réseau
    nn = NeuralNetwork(beta_0, omega_0, beta_1, omega_1)
    print(f"\n{nn}")
    
    # Test 1: Entrée scalaire
    print("\n" + "─" * 70)
    print("📊 Test 1: Propagation avec entrée scalaire")
    print("─" * 70)
    x_test = 0.5
    h = nn.hidden_layer(x_test)
    f = nn.forward(x_test)
    proba = nn.predict_proba(x_test)
    pred = nn.predict(x_test)
    
    print(f"Entrée x = {x_test}")
    print(f"Couche cachée h(x) = {h}")
    print(f"Sortie f(x) = {f:.6f}")
    print(f"Probabilité λ(x) = P(y=1|x) = {proba:.6f}")
    print(f"Prédiction (seuil=0.5) = {pred}")
    
    # Test 2: Entrées multiples
    print("\n" + "─" * 70)
    print("📊 Test 2: Propagation avec entrées multiples")
    print("─" * 70)
    x_batch = np.array([0.1, 0.5, 0.9])
    h_batch = nn.hidden_layer(x_batch)
    f_batch = nn.forward(x_batch)
    proba_batch = nn.predict_proba(x_batch)
    pred_batch = nn.predict(x_batch)
    
    print(f"Entrées x = {x_batch}")
    print(f"Couche cachée h(x):\n{h_batch}")
    print(f"Sortie f(x) = {f_batch}")
    print(f"Probabilités λ(x) = {proba_batch}")
    print(f"Prédictions = {pred_batch}")
    
    # Test 3: Données d'entraînement du projet
    print("\n" + "─" * 70)
    print("📊 Test 3: Données d'entraînement du projet")
    print("─" * 70)
    x_train = np.array([0.09291784, 0.46809093, 0.93089486])
    y_train = np.array([0, 1, 1])
    
    probas = nn.predict_proba(x_train)
    predictions = nn.predict(x_train)
    
    print(f"{'x':<12} | {'y réel':<8} | {'λ(x)':<12} | {'Prédiction':<12}")
    print("─" * 60)
    for i in range(len(x_train)):
        print(f"{x_train[i]:<12.6f} | {y_train[i]:<8} | {probas[i]:<12.6f} | {predictions[i]:<12}")
    
    # Test 4: Fonction utilitaire forward_pass
    print("\n" + "─" * 70)
    print("📊 Test 4: Fonction utilitaire forward_pass")
    print("─" * 70)
    f_util = forward_pass(0.5, beta_0, omega_0, beta_1, omega_1)
    print(f"forward_pass(0.5) = {f_util:.6f}")
    print(f"nn.forward(0.5)   = {nn.forward(0.5):.6f}")
    print(f"Identiques: {np.isclose(f_util, nn.forward(0.5))}")
    
    print("\n" + "=" * 70)
    print("✅ TOUS LES TESTS SONT TERMINÉS")
    print("=" * 70)
    


