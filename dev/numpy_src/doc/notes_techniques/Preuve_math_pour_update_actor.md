**Auteur :** Émile Jobin 

**Date :** 24 mars 2026

# 1. Preuve de la log probabilité d'une distribution normale
Premièrement, il faut commencer avec la fonction de densité d'une distribution normale ayant une moyenne $\mu$ et un écart type $\sigma$ :
$$\pi_{\theta}(a|s) = 
\prod_{i=1}^k\left(
\frac{1}{\sigma_i \sqrt{2\pi}} e^{-\frac{(a_i-\mu_i)^2}{2\sigma_i^2}}\right)
$$
> **Note** : $k$ est le nombre de sorties que nous avons


Nous cherchons donc $\log\pi_{\theta}(a|s)$
> **Note** : Dans tout ce document, $\log(x)$ est le logarithme naturel : $\ln(x)$

$$\log\pi_{\theta}(a|s) = \log\left(
\prod_{i=1}^k\left(
\frac{1}{\sigma_i \sqrt{2\pi}} e^{-\frac{(a_i-\mu_i)^2}{2\sigma_i^2}}\right)\right)
$$

Utilisons la règle des logarithmes qui dicte que :
$$\log\prod(x)=\sum(\log x)$$
Appliquons cette règle :
$$\log\pi_{\theta}(a|s) =
\sum_{i=1}^k\log\left(
    \frac{1}{\sigma_i \sqrt{2\pi}} e^{-\frac{(a_i-\mu_i)^2}{2\sigma_i^2}}\right)
$$
Nous allons utiliser une autre règle des logarithmes qui dicte que :
$$\log(xy)=\log(x)+\log(y)$$
Appliquons cette règle :

$$\log\pi_{\theta}(a|s) =
\sum_{i=1}^k\left(\log\left(
    \frac{1}{\sigma_i \sqrt{2\pi}}\right) + 
    \log\left(e^{-\frac{(a_i-\mu_i)^2}{2\sigma_i^2}}\right)\right)
$$

Les deux prochaines règles des logarithme que nous allons utiliser sont :
$$\log{\frac{1}{x}} = -\log(x) \text{ et }
\log\left(e^{x}\right) = x
$$

Appliquons ces règles :

$$\log\pi_{\theta}(a|s) =
\sum_{i=1}^k\left(-\log\left(
    \sigma_i \sqrt{2\pi}\right) -
    \frac{(a_i-\mu_i)^2}{2\sigma_i^2}\right)
$$

Utilisons donc encore nos règles d'additions :
$$\log\pi_{\theta}(a|s) =
\sum_{i=1}^k\left(-\log\left(
    \sigma_i \right) -
    \log\left((2\pi)^{\frac{1}{2}}\right) -
    \frac{(a_i-\mu_i)^2}{2\sigma_i^2}\right)
$$

Une nouvelle règle des $\log$ :
$$\log\left(x^a\right)=a\log\left(x\right)$$

Appliquons cette règle :

$$\log\pi_{\theta}(a|s) =
\sum_{i=1}^k\left(-\log\left(
    \sigma_i \right) -
    \frac{1}{2}\log(2\pi) - 
    \frac{1}{2} \frac{(a_i-\mu_i)^2}{\sigma_i^2}\right)
$$


Une nouvelle règle des sommes :
$$\sum_{i=1}^{n}ax_i=a\sum_{i=1}^{n}x_i$$

Mise en évidence simple de $\frac{-1}{2}$ et application de la règle des sommes : 
$$\log\pi_{\theta}(a|s) =
-\frac{1}{2}
\sum_{i=1}^k\left(2\log\left(
    \sigma_i \right) +
    \frac{(a_i-\mu_i)^2}{\sigma_i^2} +
    \log(2\pi) \right)
$$

Pour finir, la dernière règle des sommes :
$$\sum_{i=1}^{n}(x_i\pm a) = \sum_{i=1}^{n}(x_i) \pm an$$

Appliquons cette règle :
$$\log\pi_{\theta}(a|s) =
-\frac{1}{2} \left(
\sum_{i=1}^k
    \left(\frac{(a_i-\mu_i)^2}{\sigma_i^2} +
    2\log\left(\sigma_i \right)\right) +
    k\log(2\pi) \right)
$$
CQFD

## 1.1 Source
- Formule de la fonction de densité d'une distribution normale : Wikipedia [Loi normale](https://fr.wikipedia.org/wiki/Loi_normale)
- Pour la formule de la log prob finale : OpenAI. (2018). [*Part 1: Key Concepts in RL*](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)


# 2. Preuve des dérivées partiels de la log probabilité par rapport aux écarts type (σ) et aux moyennes (μ)
Ici, nous allons donc déterminer :
$$\frac{\partial \log\pi_{\theta}(a_i|s)}{\partial \sigma_i} \text{ et } \frac{\partial \log\pi_{\theta}(a_i|s)}{\partial \mu_i}$$

Premièrement, comme nous déterminons les dérivées partielles par rapport à un seul écart type $(\sigma_i)$ et une seule moyenne $(\mu_i)$, la somme va disparaitre de nos calculs, cela nous donne donc cette équation :

$$
\log\pi_{\theta}(a_i|s) =
-\frac{1}{2} \left(
    \frac{(a_i-\mu_i)^2}{\sigma_i^2} +
    2\log\left(\sigma_i \right) +
    \log(2\pi) \right)
$$

## 2.1 Preuve de la dérivée partiel de la log probabilité par rapport à un écart type (σ)
$$\frac{\partial \log\pi_{\theta}(a_i|s)}{\partial \sigma_i}=
\frac{\partial}{\partial \sigma_i}
\left(-\frac{1}{2} \left(
    \frac{(a_i-\mu_i)^2}{\sigma_i^2} +
2\log\left(\sigma_i \right) +
\log(2\pi) \right)\right)
\\ % sortir le -1/2
=-\frac{1}{2}
\frac{\partial}{\partial \sigma_i}
\left(
    \frac{(a_i-\mu_i)^2}{\sigma_i^2} +
2\log\left(\sigma_i \right) +
\log(2\pi) \right)
\\ % division en trois parties
= -\frac{1}{2} \left(
    \frac{\partial}{\partial \sigma_i}
    \left(
        \frac{(a_i-\mu_i)^2}{\sigma_i^2} 
    \right) +
    \frac{\partial}{\partial \sigma_i}
    \left(
        2\log\left(\sigma_i \right)
    \right) +
    \frac{\partial}{\partial \sigma_i}
    \left(
        \log(2\pi)
    \right)
\right)
$$

Isolons les trois petites dérivées :
1. $$\frac{\partial}{\partial \sigma_i}
    \left(
        \log(2\pi)
    \right) = 0$$
2. $$\frac{\partial}{\partial \sigma_i}
    \left(
        2\log\left(\sigma_i \right)
    \right) = 
    2\frac{\partial}{\partial \sigma_i}
    \left(
        \log\left(\sigma_i \right)
    \right) = 
    2\frac{1}{\sigma_i} = 
    \frac{2}{\sigma_i}
    $$
3. $$ \frac{\partial}{\partial \sigma_i}
    \left(
        \frac{(a_i-\mu_i)^2}{\sigma_i^2} 
    \right) = 
    (a_i-\mu_i)^2
    \frac{\partial}{\partial \sigma_i}
    \left(
        \frac{1}{\sigma_i^2} 
    \right) = \\
    (a_i-\mu_i)^2 \frac{-2}{\sigma_i^3} = 
    \frac{-2(a_i-\mu_i)^2}{\sigma_i^3}$$

Remettons les trois dérivées trouvé dans l'équation :
$$ =-\frac{1}{2} \left(
    \frac{\partial}{\partial \sigma_i}
    \left(
        \frac{(a_i-\mu_i)^2}{\sigma_i^2} 
    \right) +
    \frac{\partial}{\partial \sigma_i}
    \left(
        2\log\left(\sigma_i \right)
    \right) +
    \frac{\partial}{\partial \sigma_i}
    \left(
        \log(2\pi)
    \right)
\right) \\ =
-\frac{1}{2} \left(
    \frac{-2(a_i-\mu_i)^2}{\sigma_i^3} +
    \frac{2}{\sigma_i} +
    0
\right) \\ =
\frac{(a_i-\mu_i)^2}{\sigma_i^3} -
\frac{1}{\sigma_i}
$$

Donc notre équation finale est :
$$\frac{\partial \log\pi_{\theta}(a_i|s)}{\partial \sigma_i} =
\frac{(a_i-\mu_i)^2}{\sigma_i^3} -
\frac{1}{\sigma_i}
$$

## 2.1 Preuve de la dérivée partielle de la log probabilité par rapport à une moyenne (μ)
Comme avec la dérivée par rapport à un écart type, on défait la dérivée en trois petites dérivées partielles :
$$ -\frac{1}{2} \left(
    \frac{\partial}{\partial \mu_i}
    \left(
        \frac{(a_i-\mu_i)^2}{\sigma_i^2} 
    \right) +
    \frac{\partial}{\partial \mu_i}
    \left(
        2\log\left(\sigma_i \right)
    \right) +
    \frac{\partial}{\partial \mu_i}
    \left(
        \log(2\pi)
    \right)
\right)
$$

Isolons les trois petites dérivées :
1. $$\frac{\partial}{\partial \mu_i}
    \left(
        \log(2\pi)
    \right) = 0$$
2. $$\frac{\partial}{\partial \mu_i}
    \left(
        2\log\left(\sigma_i \right)
    \right) = 0
    $$
3. $$ \frac{\partial}{\partial \mu_i}
    \left(
        \frac{(a_i-\mu_i)^2}{\sigma_i^2} 
    \right) = 
    \frac{1}{\sigma_i^2}
    \frac{\partial}{\partial \mu_i}
    \left(
        (a_i-\mu_i)^2 
    \right) =

    \frac{1}{\sigma_i^2} -2(a_i-\mu_i) =
    \frac{-2(a_i-\mu_i)}{\sigma_i^2}
    $$

Remettons les trois dérivées trouvé dans l'équation :

$$ -\frac{1}{2} \left(
    \frac{\partial}{\partial \mu_i}
    \left(
        \frac{(a_i-\mu_i)^2}{\sigma_i^2} 
    \right) +
    \frac{\partial}{\partial \mu_i}
    \left(
        2\log\left(\sigma_i \right)
    \right) +
    \frac{\partial}{\partial \mu_i}
    \left(
        \log(2\pi)
    \right)
\right) \\ =
-\frac{1}{2} \left(
    \frac{-2(a_i-\mu_i)}{\sigma_i^2} + 0 + 0 
\right) \\ =
\frac{(a_i-\mu_i)}{\sigma_i^2}
$$

Donc notre équation finale est :
$$\frac{\partial \log\pi_{\theta}(a_i|s)}{\partial \mu_i} =
\frac{(a_i-\mu_i)}{\sigma_i^2}
$$