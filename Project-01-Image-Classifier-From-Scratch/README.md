# Cat Dog Panda Image Classification

Projet de classification d'images utilisant PyTorch.

## Objectif

Construire un modèle CNN capable de classifier des images en 3 classes :

- Cat
- Dog
- Panda


## Technologies

- Python
- PyTorch
- Torchvision
- Pandas
- PIL
- Matplotlib
- Seaborn
- Scikit-learn


## Dataset

Dataset composé de 3 classes :
- Cat
- Dog
- Panda


## Méthodologie

### Préprocessing

- Resize des images
- Normalisation
- Data augmentation :
  - RandomCrop
  - HorizontalFlip
  - RandomAffine
  - ColorJitter


### Modèle

CNN from scratch développé avec PyTorch.

Architecture :
- Convolution blocks
- Batch Normalization
- Dropout2d/Dropout
- MaxPool2d
- Adaptive Average Pooling
- Fully Connected layers


## Entraînement

Optimisations testées :

- Adam / AdamW
- Learning rate scheduler
- Weight decay
- Dropout
- Early stopping


## Résultats

Validation accuracy :

89 %

Kaggle test accuracy :

87.33 %


## Prediction

Le modèle génère un fichier :

submission.csv

avec le format :

ID | CLASS

