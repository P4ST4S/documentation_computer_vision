---
title: Limitations Et Travaux Futurs
---

# Limitations Et Travaux Futurs

---

## Limitations Identifiées

### Limitations Techniques

| Limitation                            | Impact          | Explication Technique                                                                                                                                                                                    |
| ------------------------------------- | --------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Instance unique par classe**        |  Majeur       | L'algorithme `mask_to_yolo_format()` sélectionne le plus grand contour uniquement (ligne 117: `max(contours, key=cv2.contourArea)`). **Exemple raté** : Image avec 2 pommes de terre → 1 seule détectée. |
| **Classes déséquilibrées**            |  Modéré       | Ratio 2.1:1 entre classe max (bread, 991 img) et min (rice, 464 img). Modèle biaisé vers classes fréquentes.                                                                                             |
| **Résolution 640×640**                |  Modéré       | Textures fines (viandes, sauces) perdent détails critiques après downscaling depuis 798×652 px moyen.                                                                                                    |
| **Pas de calibration nutritionnelle** |  Bloquant MVP | Le modèle prédit les classes mais pas les quantités (g, mL). Nécessite régression volume/poids.                                                                                                          |

### Limitations du Dataset

- **Biais géographique** : FoodSeg103 surreprésente cuisine occidentale/chinoise
- **Conditions d'acquisition** : Photos professionnelles haute qualité ≠ photos utilisateurs (flou, éclairage faible, angles extrêmes)
- **Annotations imprécises** : Erreurs manuelles dans masques (vérifiées en EDA, ~2% d'images)

## Travaux Futurs (Roadmap Technique)

### Phase 3 : Estimation Nutritionnelle (Semaine 3)

**Objectif** : Associer segmentation → quantité → calories

**Pipeline proposé** :

```python
Masque segmenté (pixels)
    │
    ▼
Estimation volume 3D
│  - Hypothèse forme (sphère, cylindre, prisme)
│  - Calibration avec objet référence (assiette standard 26 cm)
│  - Équation: Volume = f(Aire_masque, Profondeur_estimée)
    │
    ▼
Conversion poids
│  - Densité spécifique par classe (kg/L)
│  - Exemple: ρ_rice = 0.85 kg/L, ρ_ice_cream = 0.54 kg/L
    │
    ▼
Requête base nutritionnelle
│  - API: USDA FoodData Central
│  - Entrée: (classe, poids_g)
│  - Sortie: {calories, protéines, lipides, glucides}
    │
    ▼
Agrégation totale repas
```

**Défis techniques** :

1. **Estimation de profondeur monoculaire** : Réseau CNN additionnel (MiDaS, DPT) ou depth-from-stereo
2. **Calibration dynamique** : Détection automatique d'objet référence (assiette, fourchette) pour échelle

### Phase 4 : Déploiement Production (Semaine 4)

**Architecture cible** :

```
┌─────────────────┐
│  Frontend React │  (Upload image)
│  Mobile/Web     │
└────────┬────────┘
         │ HTTP POST /predict
         ▼
┌──────────────────────────────────┐
│  API REST (FastAPI)              │
│  - Endpoint: /api/v1/predict     │
│  - Rate limiting: 10 req/min     │
│  - Auth: JWT tokens              │
└────────┬─────────────────────────┘
         │
         ▼
┌──────────────────────────────────┐
│  ONNX Runtime                    │
│  - Model: best.onnx (FP16)       │
│  - Device: GPU (TensorRT) ou CPU │
│  - Latence: &lt;200ms/image         │
└────────┬─────────────────────────┘
         │
         ▼
┌──────────────────────────────────┐
│  Post-traitement                 │
│  - NMS (conf>0.25, IoU>0.45)     │
│  - Calcul nutrition par instance │
│  - Agrégation totale repas       │
└────────┬─────────────────────────┘
         │
         ▼
    JSON Response
    {
      "items": [
        {"class": "bread", "calories": 75, "weight_g": 30, "bbox": [...], "mask": [...]}
      ],
      "total_calories": 512,
      "processing_time_ms": 187
    }
```

**Stack technologique proposée** :

- **Backend** : FastAPI 0.109 (Python 3.11)
- **Inférence** : ONNX Runtime 1.17 avec TensorRT EP (NVIDIA)
- **Containerisation** : Docker (image nvidia/cuda:12.6-runtime)
- **Orchestration** : Kubernetes (scalabilité horizontale)
- **Monitoring** : Prometheus + Grafana (latence, throughput, erreurs)

### Améliorations Modèle

1. **Segmentation multi-instances** :
   - Remplacer `max(contours)` par détection de tous les contours avec filtrage par aire minimale
   - Associer un `instance_id` unique par objet

2. **Augmentation Dataset** :
   - **Style transfer** : Appliquer textures aliments sur scènes synthétiques 3D
   - **GANs** : Génération d'images de repas avec labels automatiques

3. **Architecture avancée** :
   - **Mask R-CNN** : Meilleure gestion des instances multiples (plus lent, +12% mAP attendu)
   - **SegFormer** : Transformer-based segmentation (état de l'art recherche 2025)

4. **Active Learning** :
   - Déployer en production, collecter prédictions à faible confiance (&lt;0.5)
   - Faire annotater manuellement (crowdsourcing), réentraîner périodiquement

---
