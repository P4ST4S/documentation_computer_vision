---
title: Application Web
id: application
slug: /application
---

import Icon from '@site/src/components/Icon';

# Application Web — Edge AI

> **Tester l'application** : [https://app-computer-vision.vercel.app/](https://app-computer-vision.vercel.app/)

---

## <Icon name="Cpu" size={18} /> Approche Edge AI

L'application NutriScan fonctionne **entièrement dans le navigateur** de l'utilisateur. Il n'y a **aucun serveur backend, aucune API REST, aucun conteneur Docker**. Le modèle ONNX est chargé et exécuté directement côté client grâce à ONNX Runtime Web (WebAssembly).

**Avantages de cette approche** :

- **Confidentialité** : les images ne quittent jamais l'appareil de l'utilisateur
- **Latence minimale** : pas de round-trip réseau, inférence locale
- **Scalabilité gratuite** : chaque utilisateur utilise son propre CPU/GPU
- **Déploiement simple** : fichiers statiques sur Vercel, aucune infrastructure serveur

---

## <Icon name="Layers" size={18} /> Stack Technique

| Composant       | Technologie           | Version | Role                      |
| --------------- | --------------------- | ------- | ------------------------- |
| **Framework**   | Next.js (App Router)  | 16.1    | Application React SSR/SSG |
| **UI**          | React                 | 19.2    | Interface utilisateur     |
| **Inférence**   | ONNX Runtime Web      | 1.24    | Exécution modèle WASM     |
| **Styles**      | Tailwind CSS          | 4.0     | Design responsive         |
| **Hébergement** | Vercel                | -       | Déploiement statique      |
| **Modèle**      | YOLOv8m-seg ONNX FP16 | -       | 52 MB, 12 classes         |

---

## <Icon name="GitBranch" size={18} /> Architecture Client-Side

```
┌──────────────────────────────────────────────────────────────────┐
│  Navigateur (client)                                             │
│                                                                  │
│  ┌───────────────────────────────┐                               │
│  │  Main Thread (React)          │                               │
│  │  ├─ CameraScanner.tsx         │  ◄── Capture caméra           │
│  │  ├─ ImageUploader.tsx         │  ◄── Upload fichier           │
│  │  ├─ NutritionResult.tsx       │  ◄── Affichage résultats      │
│  │  └─ useInference.ts (hook)    │                               │
│  └──────────┬────────────────────┘                               │
│             │ postMessage (base64 JPEG)                          │
│             ▼                                                    │
│  ┌───────────────────────────────┐                               │
│  │  Web Worker                   │                               │
│  │  (inference.worker.ts)        │                               │
│  │                               │                               │
│  │  1. Preprocessing             │  Image → Tensor [1,3,640,640] │
│  │  2. ONNX Runtime (WASM)       │  best.onnx (52 MB)            │
│  │  3. NMS                       │  Per-class + cross-class      │
│  │  4. Mask Generation           │  Coeffs × Prototypes          │
│  │  5. Calcul Nutrition          │  Pixels → cm² → g → kcal      │
│  └──────────┬────────────────────┘                               │
│             │ postMessage (résultats)                            │
│             ▼                                                    │
│  Affichage: card par aliment + totaux + nutrient bars            │
└──────────────────────────────────────────────────────────────────┘
```

Le **Web Worker** isole l'inférence lourde (~2-3s) du thread principal, garantissant une UI fluide à 60fps.

---

## <Icon name="FolderTree" size={18} /> Structure Du Code

```
app/src/
├── workers/
│   └── inference.worker.ts          # Web Worker ONNX Runtime
│
├── hooks/
│   ├── useInference.ts              # Hook React : cycle de vie worker
│   └── useCamera.ts                 # Hook React : accès caméra
│
├── lib/
│   ├── inference/
│   │   ├── types.ts                 # Interfaces TypeScript
│   │   ├── foodDatabase.ts          # Base nutrition 12 classes
│   │   ├── preprocessing.ts         # Image → Tensor NCHW [1,3,640,640]
│   │   ├── nms.ts                   # NMS + cross-class (confusion groups)
│   │   └── postprocessing.ts        # Masques + calcul calories
│   ├── workerClient.ts              # Communication Promise-based
│   └── constants.ts                 # Configuration inférence
│
├── components/
│   ├── CameraScanner.tsx            # Capture caméra temps réel
│   ├── ImageUploader.tsx            # Upload et preview image
│   ├── NutritionResult.tsx          # Card résultats + suppression
│   └── NutrientBar.tsx              # Barre de progression nutriment
│
└── app/
    └── page.tsx                     # Page principale (orchestration)
```

---

## <Icon name="Workflow" size={18} /> Pipeline D'Inférence

### 1. Preprocessing

- Décodage de l'image (base64 JPEG → ImageBitmap)
- Redimensionnement à 640×640 via `OffscreenCanvas`
- Conversion en tensor Float32 NCHW `[1, 3, 640, 640]` normalisé [0, 1]

### 2. Inférence ONNX

- **output0** `[1, 48, 8400]` : 4 (bbox) + 12 (classes) + 32 (mask coefficients) × 8400 anchors
- **output1** `[1, 32, 160, 160]` : 32 prototypes de masques

### 3. NMS (Non-Maximum Suppression)

- **Per-class** : suppression des boîtes de même classe avec IoU > 0.45
- **Cross-class** : suppression des classes confuses qui se chevauchent (ex: steak/porc/poulet)

### 4. Génération des masques

- Multiplication matricielle : `(1×32) @ (32×25600)` = masque 160×160
- Activation sigmoïde → binarisation (seuil 0.5) → resize 640×640

### 5. Calcul nutritionnel (physics-based)

```
pixelCount (masque binaire 640×640)
    │
    ▼ × (30/640)² cm²/pixel
areaRealCm² (surface réelle)
    │
    ▼ × épaisseur (cm, par classe)
volumeCm³
    │
    ▼ × densité (g/cm³, par classe)
weightGrams
    │
    ▼ × nutrition/100g (base de données)
{calories, protéines, glucides, lipides, fibres}
```

**Calibration** : 640 pixels = 30 cm (hypothèse assiette standard ~26 cm)

---

## <Icon name="Utensils" size={18} /> Fonctionnalités

- **Caméra temps réel** : capture et analyse instantanée
- **Upload image** : import depuis la galerie avec preview
- **Fusion automatique** : les détections de même classe sont fusionnées (ex: 3 pommes de terre = 1 entrée)
- **Suppression manuelle** : bouton X par aliment pour corriger les erreurs du modèle
- **Historique des scans** : dernières analyses conservées en session
- **Totaux nutritionnels** : calories, protéines, glucides, lipides, fibres agrégés

---

## <Icon name="Globe" size={18} /> Déploiement

L'application est déployée sur **Vercel** en tant que site Next.js statique :

- Le modèle ONNX (`best.onnx`, 52 MB) est servi depuis `/public/models/`
- Les fichiers WASM d'ONNX Runtime sont chargés depuis un CDN
- Aucune variable d'environnement serveur requise
- Déploiement automatique à chaque push sur `main`

**URL** : [https://app-computer-vision.vercel.app/](https://app-computer-vision.vercel.app/)
