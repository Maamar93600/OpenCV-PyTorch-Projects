"""
Experimental configurations tested during the project.

This file is for documentation purposes only.
The configurations below are not used directly by the training pipeline.
"""



# ============================================================
# EXPERIMENTS PERFORMED
# ============================================================

'''
Models evaluated:
    - ResNet18
    - ResNet34
    - ResNet50
    - EfficientNet-B0

Fine-tuning:
    - Fine=5 : Transfer Learning (FC only)
    - Fine=4 : Fine-Tuning (unfreeze layer4 + FC)
    - Fine=3 : Fine-Tuning (unfreeze layer3 + layer4 + FC)
    - Fine=2 : Fine-Tuning (unfreeze layer2 + layer3 + layer4 + FC)

Regularization:
    - Dropout = 0.2
    - Weight Decay = 1e-4
    - Label Smoothing = 0.1

Data Augmentation:
    - RandAugment
    - num_ops = 5
    - Magnitude = 5 and 6

Optimization:
    - Adam
    - Discriminative Learning Rates
    - SGD + Momentum was also tested

Learning Rate Scheduler:
    - CosineAnnealingLR
    - MultiStepLR was also tested

Final selected experiment:
    - ResNet50
    - Fine=2
    - Dropout=0.2
    - RandAugment (5 ops, magnitude=6)
    - Weight Decay=1e-4
    - Label Smoothing=0.1
    - Adam
    - CosineAnnealingLR
'''