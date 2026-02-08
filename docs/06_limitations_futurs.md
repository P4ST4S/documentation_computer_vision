---
title: Limitations Et Travaux Futurs
---

# Limitations Et Travaux Futurs

---

## Limitations Identifiées

### Limitations Résolues par le Modèle Fusion

Le passage au modèle Fusion (32 classes, FoodSeg103 + UEC-FoodPix) a permis de résoudre ou d'atténuer plusieurs limitations de l'ancien modèle :

| Limitation (ancien modèle) | Statut | Amélioration apportée par le modèle Fusion |
| -------------------------- | ------ | ------------------------------------------ |
| **Couverture alimentaire restreinte (12 classes)** | **Résolu** | 32 classes couvrant viandes, poissons, pâtes, pizza, hamburger, etc. |
| **Dataset limité (4 526 images)** | **Résolu** | 15 994 images (×3.5), meilleure généralisation |
| **Biais géographique** | **Atténué** | UEC-FoodPix apporte la cuisine japonaise, meilleure diversité |
| **Confusion viandes** | **Atténué** | mAP globale +10.6%, meilleure discrimination grâce à plus de données |

### Limitations Persistantes

| Limitation                     | Impact | Explication Technique                                                                                                                                                                                    |
| ------------------------------ | ------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Instance unique par classe** | Majeur | L'algorithme `mask_to_yolo_format()` sélectionne le plus grand contour uniquement (ligne 117: `max(contours, key=cv2.contourArea)`). **Exemple raté** : Image avec 2 pommes de terre → 1 seule détectée. |
| **Résolution 640×640**         | Modéré | Textures fines (viandes, sauces) perdent détails critiques après downscaling depuis 798×652 px moyen.                                                                                                    |
| **Taille du modèle Fusion**   | Modéré | 157 MB (vs 53 MB ancien modèle). Impact sur le temps de chargement dans le navigateur pour l'application Edge AI. |
| **Classes encore déséquilibrées** | Modéré | Malgré la fusion, certaines classes restent sous-représentées dans le dataset combiné. |

### Limitations du Dataset

- **Conditions d'acquisition** : Photos professionnelles haute qualité ≠ photos utilisateurs (flou, éclairage faible, angles extrêmes)
- **Annotations imprécises** : Erreurs manuelles dans masques (vérifiées en EDA, ~2% d'images)
- **Biais résiduel** : La fusion FoodSeg103 + UEC-FoodPix couvre principalement les cuisines occidentale, chinoise et japonaise

## Travaux Réalisés

### Phase 3 : Estimation Nutritionnelle (Complétée)

**Objectif** : Associer segmentation → quantité → calories

**Implémentation réalisée** (approche physics-based) :

```
Masque binaire 640×640 (pixels)
    │
    ▼ pixelCount = somme des pixels actifs
    │
    ▼ × (30/640)² cm²/pixel         ← Calibration: 640px = 30cm
areaRealCm² (surface réelle)
    │
    ▼ × épaisseur (cm)              ← Par classe (ex: pain=2cm, steak=2cm, sauce=0.5cm)
volumeCm³
    │
    ▼ × densité (g/cm³)             ← Par classe (ex: pain=0.25, steak=1.05, riz=0.75)
weightGrams
    │
    ▼ × nutrition/100g              ← Base de données intégrée (12 classes)
{calories, protéines, glucides, lipides, fibres}
    │
    ▼
Agrégation totale du repas
```

**Base de données nutritionnelle** : 12 classes avec densité, épaisseur par défaut, et valeurs nutritionnelles par 100g (calories, protéines, glucides, lipides, fibres).

### Phase 4 : Déploiement Production (Complétée)

**Architecture déployée** — Edge AI (zéro serveur) :

```
┌───────────────────────────────────────────────────────────────┐
│  Navigateur (tout côté client)                                │
│                                                               │
│  ┌────────────────────────────┐                               │
│  │  Main Thread (React)       │                               │
│  │  ├─ Caméra / Upload        │  ◄── Entrée utilisateur       │
│  │  ├─ Résultats nutrition    │  ◄── Cards + nutrient bars    │
│  │  └─ Suppression manuelle   │  ◄── Correction faux positifs │
│  └───────────┬────────────────┘                               │
│              │ postMessage                                    │
│              ▼                                                │
│  ┌────────────────────────────┐                               │
│  │  Web Worker                │                               │
│  │  ├─ ONNX Runtime WASM     │  best.onnx (52 MB, FP16)       │
│  │  ├─ NMS per + cross-class │  Conf>0.25, IoU>0.45           │
│  │  ├─ Masques segmentation  │  160×160 → 640×640             │
│  │  └─ Calcul nutritionnel   │  Pixels → cm² → g → kcal       │
│  └────────────────────────────┘                               │
└───────────────────────────────────────────────────────────────┘
         │
         │ Hébergé sur Vercel (fichiers statiques)
         │ https://app-computer-vision.vercel.app/
         ▼
┌───────────────────────────────────────────────────────────────┐
│  Vercel CDN                                                   │
│  ├─ Next.js 16 (SSG)                                          │
│  ├─ /models/best.onnx (52 MB, servi en statique)              │
│  └─ Déploiement automatique sur push main                     │
└───────────────────────────────────────────────────────────────┘
```

**Stack technique déployée** :

- **Frontend** : Next.js 16.1 + React 19.2 + Tailwind CSS 4
- **Inférence** : ONNX Runtime Web 1.24 (WebAssembly, dans Web Worker)
- **Hébergement** : Vercel (déploiement statique, CDN global)
- **Modèle** : YOLOv8m-seg ONNX FP16 (52 MB), servi depuis `/public/models/`

**Avantages de l'approche Edge AI** :

- Aucune infrastructure serveur à maintenir
- Confidentialité totale (images jamais envoyées)
- Scalabilité gratuite (chaque client exécute l'inférence)
- Latence ~2-3s (inférence locale, pas de round-trip réseau)

### Améliorations Futures

1. **Segmentation multi-instances** :
   - Remplacer `max(contours)` par détection de tous les contours avec filtrage par aire minimale
   - Associer un `instance_id` unique par objet

2. **Optimisation du modèle Fusion pour le déploiement** :
   - **Quantification INT8** : Réduire la taille du modèle de 157 MB à ~40 MB pour le déploiement Edge AI
   - **Pruning** : Élaguer les neurones peu contributifs pour accélérer l'inférence
   - **Knowledge Distillation** : Entraîner un modèle Small avec les prédictions du modèle Fusion comme "teacher"

3. **Extension du dataset fusionné** :
   - Intégrer d'autres datasets (Food-101, ISIA Food-500) pour couvrir plus de cuisines
   - **GANs** : Génération d'images de repas synthétiques pour les classes sous-représentées
   - **Active Learning** : Collecter prédictions à faible confiance en production, annoter et réentraîner

4. **Architecture avancée** :
   - **Mask R-CNN** : Meilleure gestion des instances multiples (plus lent, +12% mAP attendu)
   - **SegFormer** : Transformer-based segmentation (état de l'art recherche 2025)
   - **YOLOv9/v10** : Nouvelles architectures YOLO avec potentiel d'amélioration

5. **Calibration dynamique** :
   - Détection automatique d'objet référence (assiette, fourchette) pour calibrer l'échelle
   - Estimation de profondeur monoculaire (MiDaS) pour améliorer le calcul de volume

6. **Base de données nutritionnelle étendue** :
   - Adapter la base nutritionnelle de 12 à 32 classes pour le modèle Fusion
   - Intégrer une API nutritionnelle (Open Food Facts) pour des données plus précises

---
