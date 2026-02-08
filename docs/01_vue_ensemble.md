---
title: "Vue D'Ensemble"
slug: /
---

import Icon from '@site/src/components/Icon';

# Vue D'Ensemble

## <Icon name="Target" size={18} /> Proposition De Valeur

**NutriScan** est un système de vision par ordinateur pour la segmentation d'instances alimentaires et l'estimation nutritionnelle automatisée. Le projet implémente une pipeline complète de Deep Learning basée sur l'architecture YOLOv8-seg (Segmentation), entraînée progressivement sur le dataset FoodSeg103 puis sur un **dataset fusionné FoodSeg103 + UEC-FoodPix** couvrant **32 classes alimentaires**.

**Objectif final** : Développer un compteur de calories intelligent capable d'identifier et de quantifier les aliments à partir d'images photographiques, avec une précision au niveau pixel grâce à la segmentation d'instances.

> **Tester l'application en ligne** : [https://app-computer-vision.vercel.app/](https://app-computer-vision.vercel.app/)

---

## <Icon name="Gauge" size={18} /> Résultats Clés

### Modèle actuel : YOLOv8-Fusion (32 classes)

- **Modèle**: YOLOv8m-seg fine-tuné sur dataset fusionné
- **Performance**: mAP50 = 0.672, mAP50-95 = 0.565 (Mask)
- **Jeu de données**: FoodSeg103 + UEC-FoodPix (15 994 images, 32 classes alimentaires)
- **Temps d'entraînement**: 150 époques, batch=16

### Ancien modèle : YOLOv8m FoodSeg103 (12 classes)

- **Modèle**: YOLOv8m-seg (27.3M paramètres)
- **Performance**: mAP50 = 0.617, mAP50-95 = 0.511 (Mask)
- **Jeu de données**: FoodSeg103 (4 526 images, 12 classes alimentaires)
- **Temps d'entraînement**: ~8h20 (200 époques, RTX 2060)

**Gain du modèle Fusion** : **+8.9% mAP50** et **+10.6% mAP50-95** par rapport à l'ancien modèle, avec **2.7x plus de classes** (32 vs 12) et **3.5x plus de données** (15 994 vs 4 526 images).

---

## <Icon name="Image" size={18} /> Aperçu Visuel

| Ancien modèle (12 classes)                                           | Nouveau modèle Fusion (32 classes)                                      |
| -------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| ![Ancien modèle](/img/models/yolov8m_foodseg103/val_batch1_pred.jpg) | ![Nouveau modèle Fusion](/img/models/yolov8_fusion/val_batch1_pred.jpg) |

_Comparaison des prédictions entre l'ancien modèle (FoodSeg103, 12 classes) et le nouveau modèle Fusion (32 classes)_

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
│  Post-traitement (dans le navigateur)           │
│  - NMS per-class + cross-class                  │
│  - Masques binaires par instance (640×640)      │
│  - Scores de confiance par classe               │
└────────┬────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│  Calcul nutritionnel (Edge AI)                  │
│  - Pixels → cm² → volume → poids → kcal         │
│  - Calibration: 640px = 30cm                    │
│  - Densité et épaisseur par classe              │
└────────┬────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│  Application Web (Next.js + Vercel)             │
│  - Card par aliment détecté                     │
│  - Totaux: calories, protéines, glucides, etc.  │
│  - Suppression manuelle des faux positifs       │
└─────────────────────────────────────────────────┘
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
8. **[Application Web](application.md)**

---

## <Icon name="Clock3" size={18} /> État Actuel De Développement

- **Semaine 1-2** : Pipeline de données + Fine-tuning YOLOv8-seg sur FoodSeg103 (complété)
- **Semaine 3** : Estimation nutritionnelle physics-based (complété)
- **Semaine 4** : Application Next.js Edge AI + Déploiement Vercel (complété) — [Lien](https://app-computer-vision.vercel.app/)
- **Semaine 5** : Fusion datasets (FoodSeg103 + UEC-FoodPix) + entraînement YOLOv8-Fusion 32 classes (complété)
