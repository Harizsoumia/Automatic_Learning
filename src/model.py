"""
Architecture du réseau de neurones
Auteur : Soumia Hariz
"""
import numpy as np
from .activations import relu, sigmoid


class NeuralNetwork:
    """
    Réseau de neurones avec :
    - 1 couche cachée (3 neurones avec ReLU)
    - 1 couche de sortie (1 neurone linéaire)
    """
    
    def __init__(self, beta_0, omega_0, beta_1, omega_1):
        """
        Initialise le réseau
        
        Args:
            beta_0: biais couche cachée (array de taille 3)
            omega_0: poids couche cachée (scalaire)
            beta_1: biais sortie (scalaire)
            omega_1: poids sortie (array de taille 3)
        """
        self.beta_0 = np.array(beta_0)
        self.omega_0 = omega_0
        self.beta_1 = beta_1
        self.omega_1 = np.array(omega_1)
    
    def hidden_layer(self, x):
        """
        Calcule h(x) = ReLU(beta_0 + omega_0 * x)
        
        Args:
            x: entrée scalaire ou array
        Returns:
            h(x): array de taille 3 (si x scalaire) ou (len(x), 3) (si x array)
        """
        # Si x est un scalaire
        if np.isscalar(x):
            z = self.beta_0 + self.omega_0 * x
            return relu(z)
        
        # Si x est un array
        else:
            # Reshape pour broadcasting correct
            # x shape: (n,) -> (n, 1)
            # beta_0 shape: (3,) -> (1, 3)
            # Résultat: (n, 3)
            x = np.array(x).reshape(-1, 1)  # (n, 1)
            z = self.beta_0 + self.omega_0 * x  # Broadcasting: (n,1) + scalar * (n,1) avec (3,)
            # Correction: il faut faire autrement
            z = self.beta_0[np.newaxis, :] + self.omega_0 * x  # (1, 3) + scalar * (n, 1) = (n, 3)
            return relu(z)
    
    def forward(self, x):
        """
        Calcule f(x) = beta_1 + omega_1^T * h(x)
        
        Args:
            x: entrée scalaire ou array
        Returns:
            f(x): sortie du réseau (avant sigmoid)
        """
        # Si x est un scalaire
        if np.isscalar(x):
            h = self.hidden_layer(x)  # shape: (3,)
            f = self.beta_1 + np.dot(self.omega_1, h)  # scalaire
            return f
        
        # Si x est un array
        else:
            h = self.hidden_layer(x)  # shape: (n, 3)
            # omega_1 shape: (3,)
            # h shape: (n, 3)
            # dot product: (n, 3) @ (3,) = (n,)
            f = self.beta_1 + np.dot(h, self.omega_1)  # (n,)
            return f
    
    def predict_proba(self, x):
        """
        Calcule P(y=1|x) = λ = sigmoid(f(x))
        
        Args:
            x: entrée scalaire ou array
        Returns:
            λ (lambda): probabilité entre 0 et 1
        """
        f = self.forward(x)
        return sigmoid(f)

