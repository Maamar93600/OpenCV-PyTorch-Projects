# Classification multi-classes — Cat/Dog/Panda & plats kenyans

Sur un dataset plus proche du domaine d'ImageNet, un ResNet50 utilisé uniquement en **Transfer Learning** (backbone gelé) atteint **99 % de Validation Accuracy**, contre environ **89 %** pour le CNN entraîné **from scratch** dans le Project 01.

Ces expériences montrent que les caractéristiques du dataset cible jouent un rôle important dans le choix entre **Transfer Learning** et **Fine-Tuning**.



| Dataset / Projet           | Approche                       | Meilleure Validation Accuracy |
| ------------------------   | ------------------------------ | ----------------------------: |
| 🐱🐶🐼 Cat / Dog / Panda | CNN **From Scratch**            |                    **≈ 89 %** |
| 🐱🐶🐼 Cat / Dog / Panda | ResNet50 **Transfer Learning**  |                   **99,00 %** |
| 🍛 13 plats kenyans       | ResNet50 **Fine-Tuning**        |                   **79,34 %** |
