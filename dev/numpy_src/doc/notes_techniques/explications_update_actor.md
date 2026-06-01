# Explication de la fonction `PPOptimiser.__update_actor`
**Auteur :** Émile Jobin 

**Date :** 21 mars 2026

# Table des matières 
- [Explication de la fonction `PPOptimiser.__update_actor`](#explication-de-la-fonction-ppoptimiser__update_actor)
- [Table des matières](#table-des-matières)
- [1. La fonction](#1-la-fonction)
- [2. Contexte](#2-contexte)
  - [2.1. Symboles](#21-symboles)
- [3. Décortication](#3-décortication)
  - [3.1. Le ratio de probabilités](#31-le-ratio-de-probabilités)
      - [3.1.1. Implémentation du ratio](#311-implémentation-du-ratio)
  - [3.2. La perte non bornée](#32-la-perte-non-bornée)
    - [3.2.1. Implémentation de la perte non bornée](#321-implémentation-de-la-perte-non-bornée)
  - [3.3. La perte bornnée](#33-la-perte-bornnée)
    - [3.3.1. Implémentation de la perte bornée](#331-implémentation-de-la-perte-bornée)
- [3.4. L'entropie de l'acteur](#34-lentropie-de-lacteur)
    - [3.3.1. Implémentation de l'entropie de l'acteur](#331-implémentation-de-lentropie-de-lacteur)
- [4. Fonction de perte finale](#4-fonction-de-perte-finale)
  - [4.1 Implémentation de la fonction de perte finale](#41-implémentation-de-la-fonction-de-perte-finale)
- [5. La rétropropagation](#5-la-rétropropagation)
  - [5.1 La dérivée de la perte par rapport au ratio](#51-la-dérivée-de-la-perte-par-rapport-au-ratio)
    - [5.1.1 Implémentation de la dérivée de la perte par rapport au ratio](#511-implémentation-de-la-dérivée-de-la-perte-par-rapport-au-ratio)
  - [5.2 Trouver la dérivée de la perte par rapport à la log probabilitée de l'action.](#52-trouver-la-dérivée-de-la-perte-par-rapport-à-la-log-probabilitée-de-laction)
    - [5.2.1 La dérivée partielle du ration par rapport à notre probabilité](#521-la-dérivée-partielle-du-ration-par-rapport-à-notre-probabilité)
    - [5.2.2 La dérivée partielle de la probabilité par rapport à la log probabilité](#522-la-dérivée-partielle-de-la-probabilité-par-rapport-à-la-log-probabilité)
    - [5.2.3 Application de la dérivée en chaine](#523-application-de-la-dérivée-en-chaine)
    - [5.2.4 Implémentation de la dérivée de la perte par rapport à la log probabilitée de l'action.](#524-implémentation-de-la-dérivée-de-la-perte-par-rapport-à-la-log-probabilitée-de-laction)
  - [5.3 La dérivée de la perte par rapport aux écarts type (σ) et aux moyennes (μ)](#53-la-dérivée-de-la-perte-par-rapport-aux-écarts-type-σ-et-aux-moyennes-μ)
    - [5.3.1 La dérivée de la log probabilité par rapport aux écarts type (σ)](#531-la-dérivée-de-la-log-probabilité-par-rapport-aux-écarts-type-σ)
    - [5.3.2 La dérivée de la log probabilité par rapport aux moyennes (μ)](#532-la-dérivée-de-la-log-probabilité-par-rapport-aux-moyennes-μ)
    - [5.3.3 Implémentation de la dérivée de la perte par rapport aux écarts type (σ) et aux moyennes (μ)](#533-implémentation-de-la-dérivée-de-la-perte-par-rapport-aux-écarts-type-σ-et-aux-moyennes-μ)
  - [5.4 La dérivée de la perte par rapport aux paramètres du réseau de neurones](#54-la-dérivée-de-la-perte-par-rapport-aux-paramètres-du-réseau-de-neurones)
    - [5.4.1 Implémentation de la dérivée de la perte par rapport aux paramètres du réseau de neurones](#541-implémentation-de-la-dérivée-de-la-perte-par-rapport-aux-paramètres-du-réseau-de-neurones)
- [6. Mise à jour des paramètres](#6-mise-à-jour-des-paramètres)
- [6.1 Implémentation de la mise à jour des paramètres](#61-implémentation-de-la-mise-à-jour-des-paramètres)
- [7. Conclusion](#7-conclusion)


# 1. La fonction 
Avant toutes explications, voici la fonction en question :
```py
def __update_actor(self, 
                       obs: np.ndarray,
                       actions: np.ndarray,
                       gae: np.ndarray,
                       log_prob: np.ndarray,
                       hyper_params: HyperParams)-> float:
        self.__actor.reset_grads()
        ratios, mean, std = self.__prob_ratio(log_prob=log_prob,
                                   obs=obs,
                                   action=actions)
        surr1 = ratios * gae
        surr2 = np.clip(ratios,1-hyper_params.epsilon,1+hyper_params.epsilon) * gae
        entropy = np.sum(np.log(std) + 0.5 * LOG_2_PI + 0.5, axis=-1, keepdims=True)
        actor_loss = np.mean(-np.minimum(surr1, surr2) - hyper_params.entropy_coef * entropy)
        ratio_mask = (surr1 <= surr2)
        dratio:np.ndarray = - gae * ratio_mask
        d_log_prob:np.ndarray = dratio * ratios
        d_log_prob = d_log_prob.reshape(-1,1)
        dmean = d_log_prob * ((actions-mean)/(std**2))
        dstd = d_log_prob * ((actions-mean)**2/(std**3) - 1/std) - hyper_params.entropy_coef / std
        dmean /= obs.shape[0]
        dstd /= obs.shape[0]
        self.__actor.backward(dmean=dmean,dstd=dstd)
        self.__actor_optimizer.step(self.__actor.grads)
        return actor_loss
```
Cette fonction est située dans le module d'apprentissage par renforcement de `SimTorch` : [dev/src/SimTorch/rl/ppoptimiser.py](../../rl/ppoptimiser.py) dans la classe `PPOptimiser`

# 2. Contexte

Premièrement, le but de cette fonction.

Cette fonction se situe dans un optimiseur de réseau de neurones utilisant l'algorithme d'apprentissage par renforcement qui se nomme PPO. L'algorithme ne sera pas expliqué ici. Mais, l'important est de savoir la fonction de perte que nous souhaitons minimiser :
$$
\newcommand{\rat}{\frac{\pi_{\theta}(a|s)}{\pi_{\theta_{\text{old}}}(a|s)}}
\newcommand{\adv}{A^{\pi_{\theta_{\text{old}}}}(s,a)}

L(\theta) = - \min\left( \rat \adv , clip\left(\rat , 1-\epsilon, 1+\epsilon\right) \adv \right) - cH
$$

> **Note** : La fonction $clip$ est défini comme étant 
> $$clip(a,lim_{basse},lim_{haute})=\min(\max(a,lim_{basse}),lim_{haute})$$
> Elle sert à borner les valeurs de $a$ avec une valeur maximale possible ($lim_{haute}$) et une valeur minimale possible ($lim_{basse}$).
## 2.1. Symboles
1. $A^{\pi_{\theta_{\text{old}}}}$ : L'avantage de l'action $a$ en sachant $s$, c'est-à-dire, à quel point l'action $a$ a donné une récompense supérieure à ce qu'on s'attendait en sachant les observations $s$. En somme, si l'action $a$ a donné une meilleure récompense que prévue, l'avantage sera positif, si elle a donné une moins bonne récompense que prévus, l'avantage sera négatif et si la récompense a été la même que ce qu'on attendait, l'avantage sera nul ($0$).
2. $\pi_{\theta}(a|s)$ : Pour faire court, c'est la probabilité que $a$ soit choisi par la politique $\pi$ ayant les paramètres $\theta$ en sachant les observations $s$. (Exprimé plus en détail dans ce document [probabilité d'une distribution normale](Preuve_math_pour_update_actor.md#1-preuve-de-la-log-probabilité-dune-distribution-normale))
3. $\theta$ et $\theta_{\text{old}}$ : $\theta$ sont les paramètres présents de la politique et $\theta_{\text{old}}$ sont les vieux paramètres pour lesquel on a déterminé nos avantages $A^{\pi_{\theta_{\text{old}}}}$.
4. $\epsilon$ : C'est un nombre qui dicte à quel point la nouvelle politique peut être différente à l'ancienne politique, $\epsilon \in [0,1]$ mais, sa valeur est souvent laissée à $\epsilon = 0.2$
5. $H$ : C'est l'entropie de l'acteur, c'est-à-dire à quel point sa distribution de probabilité est dispersée, on fait que plus elle est grosse, plus l'acteur est récompensé. Le but étant de pousser l'acteur à essayer de nouvelles choses.
6. $c$ : C'est un nombre qui dicte à quel point le bonus d'entropie est grand $c \in [0,\infty)$ mais elle tourne souvent autour des $0.01$

# 3. Décortication

Cette fonction est plutôt imposante, elle va donc être divisé en plusieurs parties pour être plus digeste.

## 3.1. Le ratio de probabilités

Le ratio de probabilités ($r_{\theta}$) est ce terme dans la fonction :
$$
\providecommand{\rat}{\frac{\pi_{\theta}(a|s)}{\pi_{\theta_{\text{old}}}(a|s)}}
r_{\theta} = \rat
$$

#### 3.1.1. Implémentation du ratio
```python
ratios, mean, std = self.__prob_ratio(log_prob=log_prob,
                                      obs=obs,
                                      action=actions)
```

Dans notre implémentation $r_{\theta}$ est calculé par la fonction `__prob_ratio`. Cette fonction redonne les ratios `ratios`, mais aussi les valeurs de moyenne `mean` et d'écart type `std` que notre politique a déterminée avec les observations $s$, utile plus tard dans les calculs.

## 3.2. La perte non bornée

La perte non bornée ($\text{surr}_1$) est la partie de la fonction qui n'a pas subi de *cliping*, c'est ce terme :

$$
\providecommand{\rat}{\frac{\pi_{\theta}(a|s)}{\pi_{\theta_{\text{old}}}(a|s)}}
\providecommand{\adv}{A^{\pi_{\theta_{\text{old}}}}(s,a)}
\text{surr}_1 = \rat \adv
$$
ou, si on remplace le ratio :
$$
\providecommand{\adv}{A^{\pi_{\theta_{\text{old}}}}(s,a)}
\text{surr}_1 = r_{\theta} \adv
$$


### 3.2.1. Implémentation de la perte non bornée
```py
surr1 = ratios * gae
```
> **Note** : la variable `gae`, c'est le *generalized advatage estimation* et c'est notre avantage $A^{\pi_{\theta_{\text{old}}}}(s,a)$.

## 3.3. La perte bornnée

Cette perte-ci ($\text{surr}_2$) est la partie *clipée* de la fonction :
$$
\providecommand{\rat}{\frac{\pi_{\theta}(a|s)}{\pi_{\theta_{\text{old}}}(a|s)}}
\providecommand{\adv}{A^{\pi_{\theta_{\text{old}}}}(s,a)}
\text{surr}_2  = clip\left(\rat , 1-\epsilon, 1+\epsilon\right) \adv
$$

ou, si on remplace le ratio :

$$
\providecommand{\adv}{A^{\pi_{\theta_{\text{old}}}}(s,a)}
\text{surr}_2 = clip\left(r_{\theta} , 1-\epsilon, 1+\epsilon\right) \adv
$$



### 3.3.1. Implémentation de la perte bornée
```py
surr2 = np.clip(ratios,1-hyper_params.epsilon,1+hyper_params.epsilon) * gae
```
> **Note**: la variable `hyper_params.epsilon` représente $\epsilon$

# 3.4. L'entropie de l'acteur
L'entropie pour une distribution Gaussienne est donnée par cette équation :
$H = \sum_{i=1}^{k}\log(\sigma_i)+\frac{1}{2}\log(2\pi)+\frac{1}{2}$
> **Note** : $k$ est le nombre d'observations
### 3.3.1. Implémentation de l'entropie de l'acteur
```py
entropy = np.sum(np.log(std) + 0.5 * LOG_2_PI + 0.5, axis=-1, keepdims=True)
```
> **Note** : `LOG_2_PI` est une constante dans le code qui est égale à $\log(2\pi)$

# 4. Fonction de perte finale

Pour finir, on peut calculer la fonction de perte à l'aide des variables que nous avons déterminées :
$$L(\theta)=-\min\left(\text{surr}_1,\text{surr}_2\right)$$

## 4.1 Implémentation de la fonction de perte finale

```py
ratios, mean, std = self.__prob_ratio(log_prob=log_prob,
                                      obs=obs,
                                      action=actions)
surr1 = ratios * gae
surr2 = np.clip(ratios,1-hyper_params.epsilon,1+hyper_params.epsilon) * gae
entropy = np.sum(np.log(std) + 0.5 * LOG_2_PI + 0.5, axis=-1, keepdims=True)
actor_loss = np.mean(-np.minimum(surr1, surr2) - hyper_params.entropy_coef * entropy)
```
* `hyper_params.entropy_coef`, c'est $c$
> **Note** : Comme nos calculs sont faits en parallèle sur plusieurs pairs d'observations et d'actions, on veut trouver la perte moyenne de notre politique sur ces différents états, d'où le `np.mean()`.


# 5. La rétropropagation
Comme vous l'avez surement remarqué, ces 3-4 lignes de codes ne sont qu'une petite partie de la fonction complète. La majoritée de cette fonction sert à trouver les nouveaux paramètres qui optimiserons cette dernière fonction de perte. Pour ce faire, nous allons utiliser la rétropropagation.

## 5.1 La dérivée de la perte par rapport au ratio
Nous allons donc déterminer :

$$\frac{\partial L}{\partial r_{\theta}}$$

pour ce faire, comme nous travaillons avec des fonctions par partie ($\min$ et $clip$) nous allons trouver la dérivée partielle pour plusieurs cas distins. 

* si $\text{surr}_1$ est plus petit que $\text{surr}_2$, la dérivée de $r_{\theta}$ sera :
$$\frac{\partial L}{\partial r_{\theta}} = 
\frac{\partial (- \min(\text{surr}_1,\text{surr}_2)) }{\partial r_{\theta}}  \text{ si } \text{surr}_1 \lt \text{surr}_2 =
- \frac{\partial \text{surr}_1}{\partial r_{\theta}} = 
- \frac{\partial \left( r_{\theta} A^{\pi_{\theta_{\text{old}}}}(s,a) \right)}{\partial r_{\theta}} =
-  A^{\pi_{\theta_{\text{old}}}}(s,a) \\
\frac{\partial L}{\partial r_{\theta}} \text{ si } \text{surr}_1 \lt \text{surr}_2 = -  A^{\pi_{\theta_{\text{old}}}}(s,a)$$

* si $\text{surr}_2$ est plus petit que $\text{surr}_1$, la dérivée de $r_{\theta}$ sera : 
$$\frac{\partial L}{\partial r_{\theta}} = 
\frac{\partial (- \min(\text{surr}_1,\text{surr}_2)) }{\partial r_{\theta}}  \text{ si } \text{surr}_2 \lt \text{surr}_1 =
- \frac{\partial \text{surr}_2}{\partial r_{\theta}}=
\frac{\partial (-clip\left(r_{\theta} , 1-\epsilon, 1+\epsilon\right) A^{\pi_{\theta_{\text{old}}}}(s,a))}{\partial r_{\theta}}$$
On ne peut pas directement dériver la fonction $clip$, il faut donc encore trouver la dérivée partielle pour plusieurs cas :
* si $\text{surr}_2$ est plus petit que $\text{surr}_1$ et que $r_{\theta}$ est plus petit que $1-\epsilon$, la dérivée de $r_{\theta}$ sera : 
$$\frac{\partial L}{\partial r_{\theta}}  \text{ si } \text{surr}_2 \lt \text{surr}_1 \text{ et } r_{\theta} < 1-\epsilon \\ 
= \frac{\partial (-clip\left(r_{\theta} , 1-\epsilon, 1+\epsilon\right) A^{\pi_{\theta_{\text{old}}}}(s,a))}{\partial r_{\theta}} \text{ si } r_{\theta} < 1-\epsilon
= -\frac{\partial ((1-\epsilon)A^{\pi_{\theta_{\text{old}}}}(s,a))}{\partial r_{\theta}} = 0 \\
\frac{\partial L}{\partial r_{\theta}}  \text{ si } \text{surr}_2 \lt \text{surr}_1 \text{ et } r_{\theta} < 1-\epsilon = 0$$

* si $\text{surr}_2$ est plus petit que $\text{surr}_1$ et que $r_{\theta}$ est plus grand que $1+\epsilon$, la dérivée de $r_{\theta}$ sera : 
$$\frac{\partial L}{\partial r_{\theta}}  \text{ si } \text{surr}_2 \lt \text{surr}_1 \text{ et } r_{\theta} > 1+\epsilon \\ 
= \frac{\partial (-clip\left(r_{\theta} , 1-\epsilon, 1+\epsilon\right) A^{\pi_{\theta_{\text{old}}}}(s,a))}{\partial r_{\theta}} \text{ si } r_{\theta} > 1+\epsilon
= -\frac{\partial ((1-\epsilon)A^{\pi_{\theta_{\text{old}}}}(s,a))}{\partial r_{\theta}} = 0 \\
\frac{\partial L}{\partial r_{\theta}}  \text{ si } \text{surr}_2 \lt \text{surr}_1 \text{ et } r_{\theta} > 1+\epsilon = 0$$
* si $\text{surr}_2$ est plus petit que $\text{surr}_1$ et que $r_{\theta}$ est entre $1-\epsilon$ et $1+\epsilon$, la dérivée de $r_{\theta}$ sera : 
$$\frac{\partial L}{\partial r_{\theta}}  \text{ si } \text{surr}_2 \lt \text{surr}_1 \text{ et } 1-\epsilon \le r_{\theta} \le 1+\epsilon \\ 
= \frac{\partial (-clip\left(r_{\theta} , 1-\epsilon, 1+\epsilon\right) A^{\pi_{\theta_{\text{old}}}}(s,a))}{\partial r_{\theta}} \text{ si } 1-\epsilon \le r_{\theta} \le 1+\epsilon \\
= - \frac{\partial \left( r_{\theta} A^{\pi_{\theta_{\text{old}}}}(s,a) \right)}{\partial r_{\theta}} =
-  A^{\pi_{\theta_{\text{old}}}}(s,a)\\
\frac{\partial L}{\partial r_{\theta}}  \text{ si } \text{surr}_2 \lt \text{surr}_1 \text{ et } 1-\epsilon \le r_{\theta} \le 1+\epsilon = - A^{\pi_{\theta_{\text{old}}}}(s,a)
$$
On peut donc enfin définir la dérivée partielle de la perte par rapport au ratio comme une fonction par parti :
$$\frac{\partial L}{\partial r_{\theta}} = 
\begin{cases}
- A^{\pi_{\theta_{\text{old}}}}(s,a) & \text{si } \text{surr}_1 \lt \text{surr}_2 \\
0 & \text{si } \text{surr}_2 \lt \text{surr}_1 \text{ et } r_{\theta} < 1-\epsilon \\ 
0 & \text{si } \text{surr}_2 \lt \text{surr}_1 \text{ et } r_{\theta} > 1+\epsilon \\ 
- A^{\pi_{\theta_{\text{old}}}}(s,a) & \text{si } \text{surr}_2 \lt \text{surr}_1 \text{ et } 1-\epsilon \le r_{\theta} \le 1+\epsilon \\
\end{cases}$$

Comme $- A^{\pi_{\theta_{\text{old}}}}(s,a)$ et $0$ se répetent comme dérivées et que $\text{surr}_2 = \text{surr}_1$ si $1-\epsilon \le r_{\theta} \le 1+\epsilon$, on peut généraliser la dérivé partielle plus simplement comme étant une fonction en deux partie :
$$\frac{\partial L}{\partial r_{\theta}} = 
\begin{cases}
- A^{\pi_{\theta_{\text{old}}}}(s,a) & \text{si } \text{surr}_1 \le \text{surr}_2 \\
0 & \text{si } \text{surr}_1 \gt \text{surr}_2 \\ 
\end{cases}$$

### 5.1.1 Implémentation de la dérivée de la perte par rapport au ratio
Voici les deux lignes qui calcule notre dérivée partielle :
```py
ratio_mask = (surr1 <= surr2)
dratio:np.ndarray = - gae * ratio_mask
```
* `ratio_mask` est un masque binaire qui est $1$ si $\text{surr}_1 \le \text{surr}_2$ et $0$ si $\text{surr}_1 \gt \text{surr}_2$. Le multiplier à $- A^{\pi_{\theta_{\text{old}}}}(s,a)$ va donc nous donner $- A^{\pi_{\theta_{\text{old}}}}(s,a)$ si $\text{surr}_1 \le \text{surr}_2$ et $0$ si $\text{surr}_1 \gt \text{surr}_2$, ce qui est la fonction par partie voulut.
* `dratio` représente $\frac{\partial L}{\partial r_{\theta}}$

## 5.2 Trouver la dérivée de la perte par rapport à la log probabilitée de l'action. 
Pour un peut de contexte, en apprentissage machine, on entraine la politique à nous donner le logarithme naturel des probabilité pour plusieur raison que je ne vais pas détailler ici.

---

Nous allons donc déterminer $\frac{\partial L}{\partial \log\pi_{\theta}(a|s)}$.

Nous allons utiliser la règle de dérivée en chaine pour déterminer notre dérivée partielle. La chaine que nous allons utiliser est la suivante :
$$\frac{\partial L}{\partial \log\pi_{\theta}(a|s)} = 
\frac{\partial L}{\partial r_{\theta}} \frac{\partial r_{\theta}}{\partial \pi_{\theta}(a|s)} \frac{\partial \pi_{\theta}(a|s)}{\partial \log\pi_{\theta}(a|s)}$$
Nous avons déjà déterminé $\frac{\partial L}{\partial r_{\theta}}$ dans la section plus haut, nous allons donc déterminer les deux autre dérivées partielles, $\frac{\partial r_{\theta}}{\partial \pi_{\theta}(a|s)}$ et $\frac{\partial \pi_{\theta}(a|s)}{\partial \log\pi_{\theta}(a|s)}$

### 5.2.1 La dérivée partielle du ration par rapport à notre probabilité
La fonction de $r_{\theta}$ en fonction de notre probalilité $\pi_{\theta}(a|s)$ est, je le rappel, la suivante :
$$r_{\theta} = \frac{\pi_{\theta}(a|s)}{\pi_{\theta_{\text{old}}}(a|s)}$$

La dérivée partielle du ration $r_{\theta}$ par rapport à notre probabilité $\pi_{\theta}(a|s)$, $\frac{\partial r_{\theta}}{\partial \pi_{\theta}(a|s)}$ est donc :

$$\frac{\partial r_{\theta}}{\partial \pi_{\theta}(a|s)} = 
\frac{\partial \left(\frac{\pi_{\theta}(a|s)}{\pi_{\theta_{\text{old}}}(a|s)}\right)}{\partial \pi_{\theta}(a|s)} =
\frac{1}{\pi_{\theta_{\text{old}}}(a|s)} \frac{\partial \left(\pi_{\theta}(a|s)\right)}{\partial \pi_{\theta}(a|s)} =
\frac{1}{\pi_{\theta_{\text{old}}}(a|s)}
$$
donc
$$\frac{\partial r_{\theta}}{\partial \pi_{\theta}(a|s)} = 
\frac{1}{\pi_{\theta_{\text{old}}}(a|s)}
$$
### 5.2.2 La dérivée partielle de la probabilité par rapport à la log probabilité

La fonction de $\pi_{\theta}(a|s)$ en fonction de notre $\log$ probalilité $\log\pi_{\theta}(a|s)$ est la suivante :
$$\pi_{\theta}(a|s) = e^{\log\pi_{\theta}(a|s)}$$

La dérivée partielle de la probabilité $\pi_{\theta}(a|s)$ par rapport à la $\log$ probabilité $\log\pi_{\theta}(a|s)$, $\frac{\partial \pi_{\theta}(a|s)}{\partial \log\pi_{\theta}(a|s)}$ est donc :
$$\frac{\partial \pi_{\theta}(a|s)}{\partial \log\pi_{\theta}(a|s)} =
\frac{\partial \left(e^{\log\pi_{\theta}(a|s)} \right)}{\partial \log\pi_{\theta}(a|s)} = 
e^{\log\pi_{\theta}(a|s)} = 
\pi_{\theta}(a|s)
$$
donc
$$\frac{\partial \pi_{\theta}(a|s)}{\partial \log\pi_{\theta}(a|s)} =
\pi_{\theta}(a|s)
$$

### 5.2.3 Application de la dérivée en chaine

$$\frac{\partial L}{\partial \log\pi_{\theta}(a|s)} = 
\frac{\partial L}{\partial r_{\theta}} \frac{\partial r_{\theta}}{\partial \pi_{\theta}(a|s)} \frac{\partial \pi_{\theta}(a|s)}{\partial \log\pi_{\theta}(a|s)}=
\frac{\partial L}{\partial r_{\theta}} \frac{1}{\pi_{\theta_{\text{old}}}(a|s)} \pi_{\theta}(a|s)=
\frac{\partial L}{\partial r_{\theta}} \frac{\pi_{\theta}(a|s)}{\pi_{\theta_{\text{old}}}(a|s)}$$
Mais, $\frac{\pi_{\theta}(a|s)}{\pi_{\theta_{\text{old}}}(a|s)}$, c'est exactement la définition de $r_{\theta}$, on va donc le remplacer :
$$\frac{\partial L}{\partial \log\pi_{\theta}(a|s)} = 
\frac{\partial L}{\partial r_{\theta}} \frac{\pi_{\theta}(a|s)}{\pi_{\theta_{\text{old}}}(a|s)} = 
\frac{\partial L}{\partial r_{\theta}} r_{\theta}$$

### 5.2.4 Implémentation de la dérivée de la perte par rapport à la log probabilitée de l'action. 
```py
d_log_prob:np.ndarray = dratio * ratios
d_log_prob = d_log_prob.reshape(-1,1)
```
* `d_log_prob` représente $\frac{\partial L}{\partial \log\pi_{\theta}(a|s)}$
* `reshape(-1,1)` sert à aligner la matrice de `d_log_prob` pour pouvoir faire le reste des calculs

## 5.3 La dérivée de la perte par rapport aux écarts type (σ) et aux moyennes (μ)

Premièrement, il faut savoir que le réseau de neurones donne la moyenne ($\mu$) et l'écart type ($\sigma$) de l'action qu'il veux faire, il faut donc trouver la dérivé partielle de la perte par rapport à ces deux valeurs :
$$\frac{\partial L}{\partial \mu} \text{ et } \frac{\partial L}{\partial \sigma}$$

Pour ce faire, la dérivée en chaîne sera encore utilisé :
$$\frac{\partial L}{\partial \mu} = \frac{\partial L}{\partial \log\pi_{\sigma}(a|s)} \frac{\partial \log\pi_{\sigma}(a|s)}{\mu}\\
 \text{ et } \\
  \frac{\partial L}{\partial \alpha} = \frac{\partial L}{\partial \log\pi_{\theta}(a|s)} \frac{\partial \log\pi_{\theta}(a|s)}{\sigma}$$


La fonction de la log probabilité $\log\pi_{\theta}(a|s)$ en fonction des moyennes $\mu$ et des écart type $\sigma$ est :
$$\log\pi_{\theta}(a|s) = 
-\frac{1}{2} \left( \sum_{i=1}^{k}\left(\frac{(a_i-\mu_i)^2}{\sigma^2_i}+2\log\sigma_i\right)+k\log2 \pi \right)
$$
> **Note** : La preuve de cette formule est faite [ici](Preuve_math_pour_update_actor.md#1-preuve-de-la-log-probabilité-dune-distribution-normale)

### 5.3.1 La dérivée de la log probabilité par rapport aux écarts type (σ)

Cette dérivée partielle ne serait pas prouvée ici, mais elle est prouvée dans [ce document](Preuve_math_pour_update_actor.md#21-preuve-de-la-dérivée-partiel-de-la-log-probabilité-par-rapport-à-un-écart-type-σ). Voici la dérivée en question :
$$
\frac{\partial \log\pi_{\theta}(a_i|s)}{\partial \sigma_i} = 
\frac{(a_i-\mu_i)^2}{\sigma_i^3}-\frac{1}{\sigma_i}
$$

De plus, nous voulons aussi ajouter l'influence de l'entropie à la dérivée, pour ce faire, nous allons soustraire la dérivée de l'entropie par rapport aux écarts type à la dérivée de la log probabilité par rapport aux écarts type :
$$\frac{\partial (-cH)}{\partial \sigma_i}=
-c\frac{\partial \left(\log(\sigma_i)+\frac{1}{2}\log(2\pi)+\frac{1}{2}\right)}{\partial \sigma_i} =
-c\frac{\partial \log(\sigma_i)}{\partial \sigma_i}=-\frac{c}{\sigma_i}
$$

### 5.3.2 La dérivée de la log probabilité par rapport aux moyennes (μ)
Cette dérivée partielle ne serait pas prouvée ici, mais elle est prouvée dans [ce document](Preuve_math_pour_update_actor.md#21-preuve-de-la-dérivée-partielle-de-la-log-probabilité-par-rapport-à-une-moyenne-μ). Voici la dérivée en question :

$$
\frac{\partial \log\pi_{\theta}(a_i|s)}{\partial \mu_i} = 
\frac{a_i-\mu_i}{\sigma_i^2}
$$

### 5.3.3 Implémentation de la dérivée de la perte par rapport aux écarts type (σ) et aux moyennes (μ)
```py
dmean = d_log_prob * ((actions-mean)/(std**2))
dstd = d_log_prob * ((actions-mean)**2/(std**3) - 1/std) - hyper_params.entropy_coef / std
dmean /= obs.shape[0]
dstd /= obs.shape[0]
```
* `dmean` représente $\frac{\partial \log\pi_{\theta}(a_i|s)}{\partial \mu_i}$
* `dstd` représente $\frac{\partial \log\pi_{\theta}(a_i|s)}{\partial \sigma_i}$
* on divise les gradients par la quantité de *step* qu'on a dans notre *minibatch* pour que les gradients soient une moyenne d'un seul gradient et non de plus en plus grand si le *minibatch* est plus grand, d'où les deux lignes :  `dmean /= obs.shape[0]` et `dstd /= obs.shape[0]`

## 5.4 La dérivée de la perte par rapport aux paramètres du réseau de neurones
Cette étape n'est pas calculé dans cette fonction et ne va pas être expliqué dans ce document, puisque ce processus est déjà expliqué dans ce document : [La rétropropagation](../../../../../docs/extrants/refs/Rétropropagation.md#3-la-rétropropagation)

### 5.4.1 Implémentation de la dérivée de la perte par rapport aux paramètres du réseau de neurones
```py
self.__actor.backward(dmean=dmean,dstd=dstd)
```
* La fonction `backward` calcule toutes les dérivées partielles du réseau de neurones et elle les met dans l'attribut `.grads`

# 6. Mise à jour des paramètres
Pour finir, nous allons utiliser un optimiseur qui va prendre en entrée les gradients trouver.
# 6.1 Implémentation de la mise à jour des paramètres
```py
self.__actor_optimizer.step(self.__actor.grads)
```
* `step` fait un *pas* d'optimisation à l'aide du gradient.
# 7. Conclusion
Chaque ligne de la fonction `__update_actor` est donc enfin expliquée.

