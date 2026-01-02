"""
Architecture du réseau de neurones
Auteur : Soumia Hariz
"""
import numpy as np
from activations import relu, sigmoid


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
            h(x): array de taille 3
        """
        # TODO: Implémenter cette fonction
        # z = beta_0 + omega_0 * x
        # h = ReLU(z)
        z = self.beta_0 + self.omega_0 * x
        h = relu(z)
        return h
    
    def forward(self, x):
        """
        Calcule f(x) = beta_1 + omega_1^T * h(x)
        
        Args:
            x: entrée scalaire ou array
        Returns:
            f(x): sortie du réseau (avant sigmoid)
        """
       
        h = self.hidden_layer(x)
        
        # Si x est un scalaire, h est un vecteur (3,)
        # Si x est un array, on doit traiter chaque élément
        if np.isscalar(x):
            f = self.beta_1 + np.dot(self.omega_1, h)
        else:
            # Pour un array d'entrées, calculer pour chaque x
            f = np.array([self.beta_1 + np.dot(self.omega_1, self.hidden_layer(xi)) 
                         for xi in x])
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

# Tests
if __name__ == "__main__":
    # Paramètres du projet
    beta_0 = [0.3, -1.0, -0.5]
    omega_0 = -1.0
    beta_1 = 2.6
    omega_1 = [-24.0, -8.0, 50.0]
    
    # Créer le modèle
    model = NeuralNetwork(beta_0, omega_0, beta_1, omega_1)
    
    # Test avec x = 0.5
    x_test = 0.5
    h = model.hidden_layer(x_test)
    f = model.forward(x_test)
    prob = model.predict_proba(x_test)
    
    print(f"Test avec x = {x_test}:")
    print(f"  h(x) = {h}")
    print(f"  f(x) = {f:.4f}")
    print(f"  P(y=1|x) = {prob:.4f}")