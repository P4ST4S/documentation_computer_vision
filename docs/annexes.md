---
title: Annexes
---

## Annexes

### A. Glossaire Technique

| Terme                             | Définition                                                                                            |
| --------------------------------- | ----------------------------------------------------------------------------------------------------- |
| **mAP (mean Average Precision)**  | Métrique moyenne des précisions moyennes par classe, standard COCO pour évaluer la détection d'objets |
| **IoU (Intersection over Union)** | Ratio entre intersection et union de 2 régions, mesure l'overlap entre prédiction et vérité terrain   |
| **NMS (Non-Maximum Suppression)** | Algorithme post-traitement supprimant les détections redondantes (IoU > seuil)                        |
| **FPN (Feature Pyramid Network)** | Architecture multi-échelles fusionnant features haute/basse résolution                                |
| **Mosaic Augmentation**           | Technique combinant 4 images en une seule mosaïque pour augmenter variabilité contexte                |
| **Transfer Learning**             | Réutilisation de poids pré-entraînés (COCO) pour accélérer convergence sur nouveau dataset            |

### B. Commandes Utiles

```bash
# Vérifier GPU
nvidia-smi

# Monitoring temps réel GPU
watch -n 1 nvidia-smi

# Taille du dataset
du -sh data/raw data/processed

# Nombre d'images par split
find data/processed/images/train -name "*.jpg" | wc -l

# Lancer notebook headless (sans navigateur)
jupyter notebook --no-browser --port=8888

# Convertir notebook en script Python
jupyter nbconvert --to script notebooks/02_preprocessing.ipynb

# Exporter environnement
pip freeze > requirements_freeze.txt

# Comparer 2 runs MLflow
mlflow ui --backend-store-uri sqlite:///notebooks/mlflow.db
```

### C. Références Bibliographiques

1. **YOLOv8** : Ultralytics. (2024). _YOLOv8: Object Detection and Segmentation_. https://github.com/ultralytics/ultralytics
2. **FoodSeg103** : Wu et al. (2021). _A Large-Scale Benchmark for Food Image Segmentation_. ACM Multimedia.
3. **Albumentations** : Buslaev et al. (2020). _Albumentations: Fast and Flexible Image Augmentations_. Information, 11(2), 125.
4. **COCO Dataset** : Lin et al. (2014). _Microsoft COCO: Common Objects in Context_. ECCV 2014.
5. **ONNX** : Open Neural Network Exchange. (2024). https://onnx.ai/

### D. Contacts

**Auteur** : Antoine ROSPARS, Yann LEBIB, Jean-Pierre JANOPOULOS <br/>
**Email** : antoine.rospars@epitech.eu, yann.lebib@epitech.eu, jean-pierre.janopoulos@epitech.eu <br/>
**GitHub** : https://github.com/P4ST4S <br/>
**Date de Soutenance** : [À déterminer]

---

**Licence** : MIT License (à vérifier avec Epitech pour droits académiques)

**Dernière mise à jour** : 2026-02-06

---
