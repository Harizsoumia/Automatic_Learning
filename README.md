Voici un README concis et professionnel en français prêt à être utilisé pour votre projet. Voulez-vous que je l’ajoute directement au dépôt (création/écrasement de README.md) ? ✅

---

# Automatic_Learning 🧠

**Projet pédagogique** d’un petit réseau de neurones pour classification binaire (Auteur : Soumia Hariz).

## 🚀 Description
Ce dépôt implémente un réseau de neurones simple à une couche cachée (3 neurones, ReLU) et une sortie linéaire suivie d’une sigmoïde. Le projet illustre :
- calcul de la sortie f(x) et de λ(x)=σ(f(x)),
- calcul du Likelihood et de la Negative Log-Likelihood (NLL),
- visualisations (courbes sigmoid, Likelihood/NLL en fonction de β₁),
- scripts d’exécution, notebooks d’exploration et tests unitaires.

---

## ✅ Fonctionnalités principales
- Implémentations : model.py, activations.py, metrics.py, visualization.py 🔧  
- Scripts reproductibles : question1.py → visualisation, question2.py → likelihood, question3.py → NLL, question4.py → optimisation β₁ 📊  
- Notebooks : notebooks pour exploration et présentation interactive 📓  
- Résultats et figures automatiquement sauvegardés dans results 📁

---

## ⚙️ Prérequis & Installation
1. Cloner le dépôt :
   ```bash
   git clone <repo-url>
   cd Automatic_Learning
   ```
2. Installer les dépendances :
   ```bash
   pip install -r requirements.txt
   ```
3. Version recommandée : **Python 3.8+**

> ⚠️ Les scripts ajoutent automatiquement le dossier racine au `PYTHONPATH`, vous pouvez donc les lancer depuis le dossier racine : `python scripts/question1.py`.

---

## ▶️ Usage (exemples rapides)
- Visualiser f(x) et λ(x) :
  ```bash
  python scripts/question1.py
  ```
- Calculer le Likelihood et sauvegarder les résultats :
  ```bash
  python scripts/question2.py
  ```
- Étudier la NLL :
  ```bash
  python scripts/question3.py
  ```
- Explorer β₁ et tracer Likelihood/NLL :
  ```bash
  python scripts/question4.py
  ```

Exemple d’utilisation programmatique (extrait) :
```python
from src.model import NeuralNetwork
from src.metrics import calculate_negative_log_likelihood

beta0 = [0.3, -1.0, -0.5]
omega0 = [-1.0, 1.8, 0.65]
beta1 = 2.6
omega1 = [-24.0, -8.0, 50.0]

nn = NeuralNetwork(beta0, omega0, beta1, omega1)
f_vals = nn.forward([0.1, 0.5, 0.9])
probas = nn.predict_proba([0.1, 0.5, 0.9])
```

---

## 🧪 Tests
Exécuter les tests unitaires avec pytest :
```bash
pytest -q
```

---

## 📁 Structure du projet (rapide)
- scripts — scripts d’exercice (question1..4)  
- src — code principal : model.py, activations.py, metrics.py, visualization.py  
- notebooks — notebooks d’exploration  
- results — figures et sorties générées  
- tests — tests unitaires

---

## ✍️ Contribution
- Fork → branche → PR.  
- Respectez le style existant et ajoutez des tests pour toute fonctionnalité nouvelle.

---

## 📄 Licence & Contact
- Licence : (à préciser)  
- Auteur : Soumia Hariz — ajouter contact si souhaité.

---

Besoin d’ajustements (langue, ton, sections additionnelles, badge CI, ou création automatique du fichier README.md) ? Dites-moi et je l’applique. 🔧
