---
title: Limitations Et Travaux Futurs
---

# Limitations Et Travaux Futurs

---

## Limitations Identifiées

### Limitations Techniques

| Limitation                     | Impact | Explication Technique                                                                                                                                                                                    |
| ------------------------------ | ------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Instance unique par classe** | Majeur | L'algorithme `mask_to_yolo_format()` sélectionne le plus grand contour uniquement (ligne 117: `max(contours, key=cv2.contourArea)`). **Exemple raté** : Image avec 2 pommes de terre → 1 seule détectée. |
| **Classes déséquilibrées**     | Modéré | Ratio 2.1:1 entre classe max (bread, 991 img) et min (rice, 464 img). Modèle biaisé vers classes fréquentes.                                                                                             |
| **Résolution 640×640**         | Modéré | Textures fines (viandes, sauces) perdent détails critiques après downscaling depuis 798×652 px moyen.                                                                                                    |
| **Confusion viandes**          | Modéré | Le modèle confond fréquemment steak/porc/poulet (mAP faible sur porc ~19%). Atténué par NMS cross-class + suppression manuelle UX.                                                                       |

### Limitations du Dataset

- **Biais géographique** : FoodSeg103 surreprésente cuisine occidentale/chinoise
- **Conditions d'acquisition** : Photos professionnelles haute qualité ≠ photos utilisateurs (flou, éclairage faible, angles extrêmes)
- **Annotations imprécises** : Erreurs manuelles dans masques (vérifiées en EDA, ~2% d'images)

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

2. **Augmentation Dataset** :
   - **Style transfer** : Appliquer textures aliments sur scènes synthétiques 3D
   - **GANs** : Génération d'images de repas avec labels automatiques

3. **Architecture avancée** :
   - **Mask R-CNN** : Meilleure gestion des instances multiples (plus lent, +12% mAP attendu)
   - **SegFormer** : Transformer-based segmentation (état de l'art recherche 2025)

4. **Active Learning** :
   - Collecter prédictions à faible confiance (&lt;0.5) en production
   - Faire annoter manuellement (crowdsourcing), réentraîner périodiquement

5. **Calibration dynamique** :
   - Détection automatique d'objet référence (assiette, fourchette) pour calibrer l'échelle
   - Estimation de profondeur monoculaire (MiDaS) pour améliorer le calcul de volume

---
