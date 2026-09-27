# Segmentation sémantique d’images drone

Projet de **segmentation sémantique** réalisé dans le cadre de la compétition Kaggle **OpenCV PyTorch Segmentation Project – Round 3**.

## 🎯 Objectif

Prédire une classe pour chaque pixel d’une image drone, avec **12 classes** :

| ID | Classe       |
| -: | ------------ |
|  0 | Background   |
|  1 | Person       |
|  2 | Bike         |
|  3 | Car          |
|  4 | Drone        |
|  5 | Boat         |
|  6 | Animal       |
|  7 | Obstacle     |
|  8 | Construction |
|  9 | Vegetation   |
| 10 | Road         |
| 11 | Sky          |

## 📊 Métrique

La compétition utilise le **Dice moyen**.

Cas particulier : lorsqu'une classe est absente du masque réel et du masque prédit, le `NaN` est considéré comme **1.0** et inclus dans la moyenne sur les 12 classes.

## 🔄 Préparation des données

* Plusieurs résolutions d’entrée ont été testées :
  
       256 × 256 : premières expérimentations
       384 × 384 : expérimentations suivantes et modèle final
* Masques redimensionnés avec `INTER_NEAREST`
* Normalisation des images
* Augmentations avec **Albumentations**
* Gestion du déséquilibre des classes avec différentes pondérations

## 🧪 Expérimentations

Plusieurs configurations ont été testées :

* U-Net + ResNet50
* DeepLabV3+ + ResNet50
* DeepLabV3+ + ResNet101
* Différentes augmentations
* Différentes pondérations de classes

La pondération **1 / √fréquence** a donné les meilleurs résultats parmi les stratégies testées.

## 🏆 Meilleur modèle

**DeepLabV3+ + ResNet101**

* Image : 384 × 384
* Batch size : 8
* Learning rate : `1e-4`
* Gamma : 2
* Pondération : `1 / √fréquence`
* Meilleur Dice validation : **91,88 %**
* mIoU au meilleur Dice : **88,26 %**
* mIoU moyen sur les epochs : **85,42 %**

Ce modèle a ensuite été utilisé pour l’inférence sur le jeu de test.

## 🔎 Inférence et soumission

Pipeline :

**Image test → Resize 384×384 → Modèle → Masque prédit → Resize aux dimensions originales → RLE → CSV**

Le masque final est encodé en **Run-Length Encoding (RLE)** selon le format demandé par la compétition.

## 📈 Résultat Kaggle

Score final : **0,57689**

Le score Kaggle est inférieur aux résultats obtenus localement. Cela m’a notamment amené à approfondir les effets possibles du **redimensionnement, des détails de bordure et de la chaîne d’inférence/RLE**.

## 📚 Ce que ce projet m’a permis de travailler

* PyTorch `Dataset` / `DataLoader`
* Segmentation sémantique
* U-Net et DeepLabV3+
* Backbones ResNet
* Dice / IoU
* Déséquilibre des classes
* Data augmentation
* Inférence
* Visualisation des masques
* RLE et préparation d’une soumission Kaggle

## 🛠️ Technologies

**Python · PyTorch · OpenCV · NumPy · Albumentations · Matplotlib · TensorBoard · Pandas · Kaggle**

