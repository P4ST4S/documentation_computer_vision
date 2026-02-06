---
title: "Vue D'Ensemble"
slug: /
---

import Icon from '@site/src/components/Icon';

# Vue D'Ensemble

## <Icon name="Target" size={18} /> Proposition De Valeur

**NutriScan** est un système de vision par ordinateur pour la segmentation d'instances alimentaires et l'estimation nutritionnelle automatisée. Le projet implémente une pipeline complète de Deep Learning basée sur l'architecture YOLOv8-seg (Segmentation) appliquée au dataset académique FoodSeg103.

**Objectif final** : Développer un compteur de calories intelligent capable d'identifier et de quantifier les aliments à partir d'images photographiques, avec une précision au niveau pixel grâce à la segmentation d'instances.

---

## <Icon name="Gauge" size={18} /> Résultats Clés

- **Modèle**: YOLOv8m-seg (27.3M paramètres)
- **Performance**: mAP50 = 0.617, mAP50-95 = 0.511
- **Jeu de données**: FoodSeg103 (4526 images, 12 classes alimentaires)
- **Temps d'entraînement**: ~8h20 (200 époques, RTX 2060)

---

## <Icon name="Image" size={18} /> Aperçu Visuel

![Exemple de segmentation NutriScan](/img/models/yolov8m_foodseg103/val_batch1_pred.jpg)

_Segmentation d'instances alimentaires avec identification des classes et masques par objet_

---

## <Icon name="GitBranch" size={18} /> Flux De Données End-To-End

```
┌─────────────────┐
│  Image Brute    │ (Variable resolution: 200-5616px)
│  (RGB)          │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│  Prétraitement OpenCV/Albumentations            │
│  - Resize: 640x640 (INTER_AREA)                 │
│  - Normalisation ImageNet (μ, σ)                │
│  - Augmentations: Flip, Rotate, HSV, Dropout    │
└────────┬────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│  Modèle YOLOv8-seg                              │
│  - Backbone: CSPDarknet53                       │
│  - Neck: PAN + FPN                              │
│  - Heads: Detection + Segmentation              │
│  - Poids: Fine-tuned sur FoodSeg103 (12 classes)│
└────────┬────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│  Post-traitement                                │
│  - NMS (Non-Maximum Suppression)                │
│  - Masques binaires par instance (H×W)          │
│  - Scores de confiance par classe               │
└────────┬────────────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│  Sortie JSON    │ [{class: "bread", mask: [...], conf: 0.92}, ...]
└─────────────────┘
```

---

## <Icon name="BookOpenText" size={18} /> Parcours De Documentation

### Documentation Principale

1. **[Analyse Technique Approfondie](02_analyse_technique.md)**
2. **[Architecture Logicielle](03_architecture_logicielle.md)**
3. **[Installation Et Reproduction](04_installation.md)**
4. **[Résultats Expérimentaux](05_resultats_experimentaux.md)**
5. **[Limitations Et Travaux Futurs](06_limitations_futurs.md)**
6. **[Annexes](annexes.md)**
7. **[Galerie Des Modèles](model-gallery.md)**
8. **[API](api.md)**

---

## <Icon name="Clock3" size={18} /> État Actuel De Développement

-  **Semaine 1-2** : Pipeline de données + Fine-tuning YOLOv8-seg (complété)
-  **Semaine 3** : Régression nutritionnelle (calibration volume/poids)
-  **Semaine 4** : API REST (FastAPI) + Dockerisation
