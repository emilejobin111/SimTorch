## Pour toutes les méthodes :

### Customisation des optimiseurs :

#### SGD (Stochastic Gradient Descent) :
1. **taux d'apprentissage / alpha** (`float`) : *default* = 0.001, *min* = 0.00001, *max* = 1.0. La taille du pas (learning rate) appliqué dans la direction du gradient lors de la mise à jour des paramètres.

---

#### Adam (Adaptive Moment Estimation) :
1. **taux d'apprentissage / alpha** (`float`) : *default* = 0.01, *min* = 0.00001, *max* = 1.0. Le taux d'apprentissage de base.
2. **beta1** (`float`) : *default* = 0.9, *min* = 0.0, *max* = 0.9999. Taux de décroissance exponentielle pour l'estimation du premier moment (moyenne des gradients passés).
3. **beta2** (`float`) : *default* = 0.999, *min* = 0.0, *max* = 0.9999. Taux de décroissance exponentielle pour l'estimation du second moment (moyenne des gradients au carré passés).
4. **epsilon** (`float`) : *default* = 1e-8, *min* = 1e-10, *max* = 1e-5. Une très petite constante ajoutée au dénominateur pour assurer la stabilité numérique (évite les divisions par zéro).

---

#### AdamW (Adam with Weight Decay) :
1. **taux d'apprentissage / alpha** (`float`) : *default* = 0.01, *min* = 0.00001, *max* = 1.0. Le taux d'apprentissage de base.
2. **décroissance des poids / weight decay** (`float`) : *default* = 0.0, *min* = 0.0, *max* = 1.0. Coefficient de pénalité appliqué directement sur les poids (découplé du gradient) pour limiter leur explosion et améliorer la généralisation du modèle.
3. **écrêtage du gradient / grad clip** (`float`) : *default* = inf, *min* = 0.1, *max* = 100.0. Norme maximale autorisée pour le gradient. Si le gradient dépasse cette valeur, il est réduit proportionnellement (clipping) pour éviter l'instabilité de l'entraînement. *(Note : si ton interface ne gère pas l'infini, tu peux mettre une valeur par défaut élevée comme 10.0 ou utiliser une case à cocher pour l'activer/désactiver).*
4. **beta1** (`float`) : *default* = 0.9, *min* = 0.0, *max* = 0.9999. Taux de décroissance exponentielle pour le premier moment.
5. **beta2** (`float`) : *default* = 0.999, *min* = 0.0, *max* = 0.9999. Taux de décroissance exponentielle pour le second moment.
6. **epsilon** (`float`) : *default* = 1e-8, *min* = 1e-10, *max* = 1e-5. Petite constante pour la stabilité numérique.


## Hyper params commun de toute les méthodes :
1. **nombre d'époques** (`int`) : *default* = 100, *min* = 1, *max* = 1000. Nombre d'itérations d'entraînement globales.
2. **normalisation des observations** (`bool`) : *default* = true. Active ou désactive la normalisation des états de l'environnement (entrée : case à cocher).

---

## Pour OpenAIES :
### juste un seul optimiseur : "paramètres pour l'Optimiseur du réseau principale"

3. **nombre de mutants** (`int`) : *default* = 50, *min* = 1, *max* = 1000. Taille de la population générée à chaque itération.
4. **nombre d'épisodes** (`int`) : *default* = 1, *min* = 1, *max* = 10. Nombre d'épisodes joués par mutant pour évaluer sa performance.
5. **sigma** (`float`) : *default* = 0.05, *min* = 0.001, *max* = 0.5. Bruit standard ajouté aux poids pour générer les mutants (taux de mutation).
6. **mutants miroirs** (`bool`) : *default* = false. Génère des paires de mutants opposés (+epsilon et -epsilon) pour réduire la variance du gradient (entrée : case à cocher).
7. **objectif** (`float`) : *default* = None, *min* = -10000.0, *max* = 10000.0. Score cible (récompense) à atteindre pour arrêter l'entraînement prématurément.

---

## Pour PPO :
### deux optimiseur : "paramètre pour l'optimiseur du réseau acteur", "paramètre pour l'optimiseur du réseau critique"

3. **longueur du déploiement** (`int`) : *default* = 1024, *min* = 128, *max* = 8192. Nombre de pas de temps collectés par environnement avant d'effectuer une mise à jour.
4. **gamma** (`float`) : *default* = 0.99, *min* = 0.8, *max* = 0.9999. Facteur d'escompte accordant plus ou moins d'importance aux récompenses futures.
5. **lambda** (`float`) : *default* = 0.90, *min* = 0.8, *max* = 1.0. Paramètre de lissage pour l'estimation de l'avantage généralisé (GAE), équilibrant biais et variance.
6. **époques d'optimisation** (`int`) : *default* = 5, *min* = 1, *max* = 30. Nombre d'itérations de descente de gradient sur le même lot de données collectées.
7. **taille du mini-lot** (`int`) : *default* = 64, *min* = 8, *max* = 4096. Taille des sous-lots divisant les données collectées lors de l'optimisation.
8. **epsilon** (`float`) : *default* = 0.2, *min* = 0.05, *max* = 0.5. Marge d'écrêtage (clipping) qui empêche la politique de subir de trop grandes variations entre deux mises à jour.
9. **coefficient d'entropie** (`float`) : *default* = 0.0, *min* = 0.0, *max* = 0.1. Coefficient ajouté à la fonction de perte pour encourager l'agent à explorer d'autres actions.

---

## Pour SAC :
### deux optimiseurs : "paramètres pour l'optimiseur du réseau pi", "paramètres pour l'optimiseur des fonctions Q"



3. **durée d'exploration initiale** (`int`) : *default* = 10000, *min* = 1000, *max* = 100000. Nombre de pas initiaux joués avec des actions purement aléatoires pour pré-remplir la mémoire.
4. **taille du buffer** (`int`) : *default* = 1000000, *min* = 10000, *max* = 10000000. Taille maximale du replay buffer stockant l'historique des transitions de l'agent.
5. **pas par itération** (`int`) : *default* = 256, *min* = 1, *max* = 10000. Nombre de pas effectués dans l'environnement entre chaque phase de mise à jour du réseau.
6. **taille du lot** (`int`) : *default* = 256, *min* = 32, *max* = 2048. Taille du lot d'expériences tiré aléatoirement du replay buffer pour entraîner les réseaux.
7. **gamma** (`float`) : *default* = 0.95, *min* = 0.8, *max* = 0.9999. Facteur d'escompte pour le calcul de la valeur Q cible.
8. **log alpha** (`float`) : *default* = 0.0, *min* = -10.0, *max* = 5.0. Valeur initiale du logarithme de la température (régule le compromis entre exploitation et exploration).
9. **tau** (`float`) : *default* = 0.005, *min* = 0.001, *max* = 0.1. Facteur d'interpolation pour la mise à jour lente et stable des réseaux cibles (polyak averaging).
10. **entropie cible** (`float`) : *default* = -1.0, *min* = -100.0, *max* = 0.0. Valeur manuelle de l'entropie cible à maintenir (utilisée uniquement si l'option automatique est désactivée).
11. **entropie cible auto** (`bool`) : *default* = true. Définit automatiquement l'entropie cible selon l'opposé de la dimension de l'espace d'action : `-dim(A)` (entrée : case à cocher).
12. **alpha entraînable** (`bool`) : *default* = true. Active l'ajustement automatique de la température d'entropie (alpha) pendant l'entraînement (entrée : case à cocher).
13. **fréquence d'évaluation** (`int`) : *default* = 5, *min* = 1, *max* = 100. Fréquence (en époques) à laquelle l'agent est évalué sur l'environnement avec des actions purement déterministes.   