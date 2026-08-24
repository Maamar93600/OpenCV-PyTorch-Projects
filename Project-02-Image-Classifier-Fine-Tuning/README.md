# Classification multi-classes de plats kenyans

## 🎯 Objectif et jeu de données

L'objectif de ce projet était de développer un modèle capable de classer automatiquement une image parmi 13 catégories de plats kenyans.

Le jeu de données utilisé provient de la compétition Kaggle OpenCV – PyTorch Project 2 – Classification.

**Kaggle — OpenCV PyTorch Project-2 Classification - Round 4**
[Voir la compétition sur Kaggle](https://www.kaggle.com/competitions/open-cv-py-torch-project-2-classification-round-4/overview)

Le dataset contient 13 classes de plats kenyans, avec des catégories parfois visuellement très proches. Certaines classes peuvent partager des caractéristiques telles que la texture, la couleur, la forme ou l'apparence générale du plat.

Proximité avec ImageNet

Le modèle utilisé, ResNet50, est initialement pré-entraîné sur ImageNet.

ImageNet constitue un très vaste jeu de données généraliste contenant de nombreuses catégories d'objets et d'images naturelles. Notre dataset de plats kenyans appartient donc au même domaine général des images naturelles, mais le problème de classification est sensiblement différent.

Les premières caractéristiques apprises par le réseau sur ImageNet, comme les contours, textures, couleurs et motifs, peuvent rester utiles pour notre problème. En revanche, les représentations plus complexes et de haut niveau ont été apprises pour reconnaître les catégories présentes dans ImageNet et ne sont pas spécifiquement adaptées à la distinction entre différents plats.

Cette différence de domaine rend le Fine-Tuning particulièrement intéressant : plutôt que de conserver toutes les représentations apprises sur ImageNet, nous cherchons à adapter une partie du backbone aux caractéristiques spécifiques des plats kenyans.
---

## 1. Du Transfer Learning au Fine-Tuning

Une première approche a consisté à utiliser le modèle pré-entraîné sur **ImageNet** en Transfer Learning, en conservant principalement les représentations du backbone et en entraînant la couche de classification finale.

Cependant, les performances obtenues ont montré que cette approche était limitée pour notre problème.

Nous nous sommes donc orientés vers le Fine-Tuning, qui permet de dégeler progressivement certaines couches du backbone afin d'adapter les représentations apprises aux caractéristiques spécifiques de notre dataset.

La question principale est alors devenue :

Quelle profondeur de Fine-Tuning permet d'obtenir le meilleur compromis entre adaptation et généralisation ?

Autrement dit,Le modèle doit suffisamment s'adapter à notre dataset pour apprendre les caractéristiques spécifiques des plats,sans simplement mémoriser les images d'entraînement. Il doit être capable de réutiliser ce qu'il a appris sur des images qu'il n'a jamais vues.

Dégeler davantage de couches augmente le nombre de paramètres entraînables et donc la capacité d'adaptation du modèle. Cependant, cette capacité supplémentaire peut également favoriser le surapprentissage.

Nous avons donc procédé progressivement afin d'étudier expérimentalement l'influence du nombre de couches dégelées.

---

## 2. Quelle profondeur de Fine-Tuning ?

Le Fine-Tuning consiste à dégeler certaines couches du backbone afin de permettre au réseau de modifier les représentations apprises lors du pré-entraînement.

Une question importante s'est alors posée :

> **Combien de couches devons-nous dégeler ?**

Dégeler davantage de couches augmente le nombre de paramètres entraînables et donc la capacité d'adaptation du modèle. Cependant, cette capacité supplémentaire peut également augmenter le risque de surapprentissage, notamment lorsque le modèle dispose de nombreuses possibilités pour s'adapter aux données d'entraînement.

Nous avons donc procédé progressivement afin d'observer expérimentalement l'influence de la profondeur du Fine-Tuning.

---

## 3. Fine-Tuning et régularisation

Avec une configuration de Fine-Tuning plus importante (**Fine=3**), les performances se sont améliorées et nous avons obtenu une accuracy de validation maximale de :

**78,8064 %**

Parallèlement, plusieurs techniques permettant de contrôler le surapprentissage ont été mises en place :

* Dropout
* Weight Decay
* Label Smoothing
* RandAugment
* Learning rates discriminants selon les couches
* CosineAnnealingLR

L'objectif était de permettre au modèle d'avoir suffisamment de capacité pour adapter ses représentations tout en conservant une bonne capacité de généralisation.

---

## 4. Analyse des erreurs

L'analyse de la matrice de confusion a permis d'identifier plusieurs classes particulièrement difficiles.

Le modèle éprouve notamment des difficultés à distinguer :

* **kukuchoma ↔ nyamachoma**
* **sukumawiki ↔ mukimo ↔ ugali**

Ces confusions peuvent s'expliquer par certaines caractéristiques visuelles communes entre ces plats, notamment leur **texture, leur couleur ou leur apparence générale**.

Cette analyse a permis d'orienter les expérimentations suivantes : il ne s'agissait plus uniquement de chercher à augmenter l'accuracy globale, mais également d'améliorer la capacité du modèle à distinguer les classes les plus difficiles.

---

## 5. Influence de RandAugment

Afin d'améliorer la généralisation, nous avons augmenté la magnitude de **RandAugment de 5 à 6**, tout en conservant la configuration Fine=3.

Cette modification a permis d'obtenir une nouvelle amélioration :

**Val Accuracy maximale : 79,3420 %**

Cette configuration constitue la meilleure performance de validation obtenue avec notre ResNet50 dans les expériences principales.

L'augmentation de la diversité des images présentées pendant l'entraînement semble avoir contribué à améliorer la capacité de généralisation du modèle.

---

## 6. Expérience supplémentaire : Fine=2 + RandAugment M6

Une dernière expérience a été réalisée par curiosité : **dégeler une couche supplémentaire**, en passant à **Fine=2**, tout en conservant RandAugment avec une magnitude de 6.

Cette expérience était particulièrement intéressante car elle augmentait encore le nombre de paramètres entraînables.

On pouvait donc s'attendre à une augmentation du risque de surapprentissage.

Pourtant, les résultats ont montré un comportement différent.

| Configuration          | Val Accuracy max | Gap au meilleur epoch |
| ---------------------- | ---------------: | --------------------: |
| ResNet50 + Fine=3 + M6 |    **79,3420 %** |          **≈ 14,3 %** |
| ResNet50 + Fine=2 + M6 |    **78,5004 %** |           **≈ 5,0 %** |

L'accuracy maximale de validation est légèrement inférieure avec Fine=2, mais le gap entre l'entraînement et la validation est fortement réduit.

Cela suggère que la combinaison :

**Fine-Tuning plus profond + RandAugment M6 + régularisation**

a pu permettre un meilleur équilibre entre **capacité d'adaptation et généralisation**.

Il s'agit d'une hypothèse expérimentale : le faible gap ne permet pas à lui seul de prouver une meilleure généralisation. Cependant, un élément supplémentaire vient renforcer cette interprétation.

---

## 7. Validation sur le jeu de test Kaggle

La configuration Fine=2 + RandAugment M6 a également obtenu de meilleurs résultats sur le jeu de test Kaggle.

Le score est passé de :

**0,74726 → 0,75577**

Cette progression constitue mon **nouveau meilleur score personnel** sur la compétition.

Ce résultat est particulièrement intéressant car il montre que l'amélioration observée ne se limite pas aux données de validation : elle se retrouve également sur un jeu de données de test externe.

---

## 🧠 Conclusion

Cette série d'expérimentations montre qu'il ne suffit pas d'utiliser une architecture plus complexe ou de dégeler davantage de couches.

La performance dépend de l'équilibre entre plusieurs éléments :

**Architecture → Fine-Tuning → Learning Rate → Data Augmentation → Régularisation → Généralisation**

L'expérience Fine=2 + M6 est particulièrement intéressante car elle montre qu'une augmentation de la capacité du modèle ne conduit pas nécessairement à davantage de surapprentissage.

Dans notre cas, la combinaison d'un Fine-Tuning plus profond et d'une forte augmentation des données a produit un modèle avec une accuracy de validation légèrement inférieure, mais présentant un gap beaucoup plus faible et surtout **un meilleur score sur Kaggle**.

Cette expérimentation illustre ainsi l'importance de ne pas considérer uniquement l'accuracy comme métrique de décision, mais également d'analyser **le comportement du modèle, les classes difficiles, la matrice de confusion, le gap entraînement/validation et les performances sur des données non vues**.
