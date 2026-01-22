"""
Package pour le projet de classification binaire et entropie croisée.
Regroupe les modules: activations, metrics, model, visualization
"""

from .activations import relu, sigmoid
from .metrics import calculate_likelihood, calculate_negative_log_likelihood
from .model import forward_pass
from .visualization import plot_sigmoid_with_data, plot_likelihood_vs_beta1

__all__ = [
    'relu',
    'sigmoid',
    'calculate_likelihood',
    'calculate_negative_log_likelihood',
    'forward_pass',
    'plot_sigmoid_with_data',
    'plot_likelihood_vs_beta1'
]
