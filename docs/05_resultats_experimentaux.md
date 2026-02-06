---
title: Résultats Expérimentaux
---

# Résultats Expérimentaux

### Métriques Globales

| Modèle           | Paramètres | mAP50 (Box) | mAP50-95 (Box) | mAP50 (Mask) | mAP50-95 (Mask) | Taille | FPS (RTX 2060) |
| ---------------- | ---------- | ----------- | -------------- | ------------ | --------------- | ------ | -------------- |
| **YOLOv8s-seg**  | 11.8M      | 0.586       | 0.496          | **0.587**    | **0.475**       | 23 MB  | ~45            |
| **YOLOv8m-seg**  | 27.3M      | 0.620       | 0.541          | **0.617**    | **0.511**       | 53 MB  | ~32            |
| **Amélioration** | +131%      | +5.8%       | +9.1%          | **+5.1%**    | **+7.6%**       | +130%  | -29%           |

**Observations** :

- YOLOv8m surpasse YOLOv8s sur toutes les métriques (gain moyen +7%)
- Le ratio paramètres/performance n'est pas linéaire (rendements décroissants)
- La segmentation (Mask) bénéficie plus du modèle Medium que la détection (Box)

#### Comparaison Visuelle YOLOv8s vs YOLOv8m

| YOLOv8s-seg | YOLOv8m-seg |
| --- | --- |
| ![YOLOv8s pred](/img/models/yolov8s_foodseg103/val_batch0_pred.jpg) | ![YOLOv8m pred](/img/models/yolov8m_foodseg103/val_batch0_pred.jpg) |

_Comparaison directe sur un batch de validation avec le meme type de scene._

### Performance par Classe (YOLOv8m-seg)

#### Top 3 Classes (mAP50-95 Mask)

| Classe          | mAP50 | mAP50-95 | Précision | Recall | Images |
| --------------- | ----- | -------- | --------- | ------ | ------ |
| **french_bean** | 0.951 | 0.883    | 0.928     | 0.894  | 704    |
| **onion**       | 0.845 | 0.754    | 0.812     | 0.798  | 881    |
| **rice**        | 0.823 | 0.751    | 0.789     | 0.801  | 464    |

**Analyse** : Classes avec formes géométriques simples (french_bean = cylindres, onion = sphères/rondelles) sont mieux segmentées.

#### Bottom 3 Classes (mAP50-95 Mask)

| Classe           | mAP50 | mAP50-95 | Précision | Recall | Images |
| ---------------- | ----- | -------- | --------- | ------ | ------ |
| **pork**         | 0.412 | 0.168    | 0.523     | 0.389  | 474    |
| **chicken_duck** | 0.534 | 0.289    | 0.612     | 0.507  | 848    |
| **sauce**        | 0.589 | 0.309    | 0.648     | 0.556  | 818    |

**Hypothèses explicatives** :

1. **Texture complexe** : La viande (pork, chicken_duck) a une texture hétérogène (fibres, graisses, cuisson variable)
2. **Formes irrégulières** : Sauce liquide sans contours définis
3. **Confusions inter-classes** : Poulet vs porc (similarité visuelle haute)

### Distribution des Erreurs

**Matrice de confusion (YOLOv8m, test set)** :

![Matrice de confusion](/img/models/yolov8m_foodseg103/confusion_matrix.png)

_Figure 1 : Matrice de confusion du modèle YOLOv8m-seg sur le test set_

![Matrice de confusion normalisée](/img/models/yolov8m_foodseg103/confusion_matrix_normalized.png)

_Figure 2 : Matrice de confusion normalisée (pourcentages)_

**Top confusions identifiées** :

```
Vraie classe vs Prédiction (top confusions)

chicken_duck ─► pork (34 erreurs)         # Textures similaires
steak ─► pork (28 erreurs)                # Viandes rouges confondues
sauce ─► background (21 erreurs)          # Zones liquides non détectées
mixed_vegetables ─► french_bean (18 erreurs)  # Légumes verts agrégés
```

**Cause principale** : Manque de features discriminantes pour textures fines (résolution 640×640 insuffisante pour détails micro-texturaux).

### Courbes d'Apprentissage

**Évolution des métriques sur 200 époques** :

![Courbes d'apprentissage](/img/models/yolov8m_foodseg103/results.png)

_Figure 3 : Évolution des métriques d'entraînement et de validation (losses, mAP, précision, recall) sur 200 époques_

**Observations clés** :

- Convergence rapide (0-50 époques) : Transfer learning efficace depuis COCO
- Plateau (150-200 époques) : Modèle atteint capacité maximale pour ce dataset
- Pas d'overfitting visible : Gap train/val &lt;3% (régularisation augmentations efficace)
- Best epoch: 187 (mAP50=0.6250)
- Early stopping: Non déclenché (patience=50, dégradation max=13 époques)

### Courbes de Performance (Segmentation)

**Courbe Précision-Recall (Mask)** :

![PR Curve Mask](/img/models/yolov8m_foodseg103/MaskPR_curve.png)

_Figure 4 : Courbe Précision-Recall pour la segmentation par classe_

**Courbe F1-Score (Mask)** :

![F1 Curve Mask](/img/models/yolov8m_foodseg103/MaskF1_curve.png)

_Figure 5 : Courbe F1-Score en fonction du seuil de confiance_

**Courbes Précision et Recall individuelles** :

![P Curve Mask](/img/models/yolov8m_foodseg103/MaskP_curve.png)

_Figure 6 : Courbe de précision par classe_

![R Curve Mask](/img/models/yolov8m_foodseg103/MaskR_curve.png)

_Figure 7 : Courbe de recall par classe_

### Exemples Visuels de Prédictions

**Comparaison Labels vs Prédictions (Batch de validation)** :

![Validation Batch 0 - Labels](/img/models/yolov8m_foodseg103/val_batch0_labels.jpg)

_Figure 8 : Ground truth (annotations réelles) - Batch de validation 0_

![Validation Batch 0 - Prédictions](/img/models/yolov8m_foodseg103/val_batch0_pred.jpg)

_Figure 9 : Prédictions du modèle - Batch de validation 0_

**Observations visuelles** :

- Les masques de segmentation prédits sont généralement bien alignés avec les ground truth
- Certaines classes comme french_bean et onion montrent une excellente qualité de segmentation
- Les classes avec textures complexes (pork, chicken_duck) présentent des contours moins précis
- Le modèle gère bien les scènes multi-objets avec plusieurs aliments par image

---
