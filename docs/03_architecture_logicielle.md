---
title: Architecture Logicielle
---

# Architecture Logicielle

### Structure Arborescente Complète

```
G-AIA-910-PAR-9-2-computervision-12/
│
├── data/                           # Données brutes et prétraitées
│   ├── raw/                        # (Non versionné, .gitignore)
│   │   └── foodseg103/            # Dataset HuggingFace (4.5 GB)
│   │       ├── train/             # 4983 images (split HF)
│   │       │   ├── data-00000-of-00002.arrow
│   │       │   └── data-00001-of-00002.arrow
│   │       └── validation/        # (Non utilisé, split custom préféré)
│   └── processed/                 # Format YOLO (versionné)
│       ├── images/
│       │   ├── train/             # 3168 images (640×640 JPEG)
│       │   ├── val/               # 679 images
│       │   └── test/              # 679 images
│       ├── labels/
│       │   ├── train/             # 2880 fichiers .txt (polygones)
│       │   ├── val/               # 617 fichiers .txt
│       │   └── test/              # 618 fichiers .txt
│       └── dataset.yaml           # Configuration YOLO (12 classes)
│
├── notebooks/                     # Pipeline Jupyter interactif
│   ├── 01_data_exploration.ipynb         # EDA (1.98 MB, 103 classes analysées)
│   ├── 02_preprocessing.ipynb            # Prétraitement (1.22 MB, filtrage 12 classes)
│   ├── 03_baseline_yolov8.ipynb          # YOLOv8s-seg (145 KB, mAP50=0.587)
│   ├── 03_baseline_yolov8_medium.ipynb   # YOLOv8m-seg (152 KB, mAP50=0.617)
│   ├── 04_quantization_export.ipynb      # Export ONNX (3 KB)
│   ├── mlflow.db                          # SQLite tracking (expériences)
│   ├── yolov8s-seg.pt                    # Poids pré-entraînés Small (23 MB)
│   └── yolov8m-seg.pt                    # Poids pré-entraînés Medium (54 MB)
│
├── src/                           # Code source modulaire
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── download.py            # Téléchargement FoodSeg103 (HuggingFace)
│   │   └── preprocessing.py       # Classe FoodSegPreprocessor (202 lignes)
│   │       # - mask_to_yolo_format() : Algorithme de conversion
│   │       # - Albumentations pipeline
│   ├── models/
│   │   ├── __init__.py
│   │   └── yolov8_trainer.py      # Wrapper YOLOv8 + MLflow (131 lignes)
│   └── utils/
│       ├── __init__.py
│       ├── metrics.py             # IoU, Dice, mIoU, Pixel Accuracy (123 lignes)
│       └── visualization.py       # Overlays, grids, heatmaps (149 lignes)
│
├── models/                        # Checkpoints entraînés
│   ├── yolov8s_foodseg103/
│   │   ├── args.yaml              # Hyperparamètres d'entraînement
│   │   ├── weights/
│   │   │   ├── best.pt            # Meilleur modèle (mAP50 val max)
│   │   │   ├── last.pt            # Dernière époque
│   │   │   └── epoch*.pt          # Checkpoints périodiques (tous les 15 époques)
│   │   ├── results.csv            # Métriques par époque
│   │   ├── results.png            # Courbes d'apprentissage
│   │   ├── confusion_matrix.png
│   │   ├── F1_curve.png
│   │   ├── P_curve.png            # Precision curve
│   │   ├── R_curve.png            # Recall curve
│   │   ├── PR_curve.png           # Precision-Recall curve
│   │   └── val_batch*.jpg         # Prédictions validation
│   └── yolov8m_foodseg103/        # (Structure identique)
│       └── weights/
│           ├── best.pt            # 52.3 MB PyTorch
│           └── best.onnx          # 52.1 MB ONNX FP16
│
├── scripts/                       # Utilitaires système
│   ├── download_datasets.sh      # Bash (Linux/macOS)
│   └── download_datasets.ps1     # PowerShell (Windows)
│
├── fix_labels.py                  # Script de correction labels (27 lignes)
├── requirements.txt               # Dépendances Python (32 lignes)
├── README.md                      # Documentation utilisateur
├── DOCS.md                        # (Ce fichier) Documentation technique
└── .gitignore                     # Exclusions Git (data/raw/, *.pt, mlruns/)
```

### Modules Python : Diagramme de Dépendances

```
┌──────────────────────────────────────────────────────────┐
│  notebooks/*.ipynb (Interface utilisateur)               │
│  - Protocole expérimental                                │
│  - Visualisations                                        │
│  - Tracking MLflow                                       │
└─────────────────┬────────────────────────────────────────┘
                  │
                  ├─────► src.data.download
                  │       - download_foodseg103()
                  │       - verify_dataset_integrity()
                  │
                  ├─────► src.data.preprocessing
                  │       - FoodSegPreprocessor
                  │       - mask_to_yolo_format()
                  │       - create_yolo_dataset_yaml()
                  │
                  ├─────► src.models.yolov8_trainer
                  │       - YOLOv8Trainer
                  │       - train() [MLflow logging]
                  │       - validate()
                  │       - predict()
                  │
                  └─────► src.utils.*
                          - metrics: IoU, Dice, mIoU
                          - visualization: Matplotlib overlays

External Dependencies:
  - ultralytics.YOLO (YOLOv8 core)
  - albumentations (transformations)
  - mlflow (tracking)
  - cv2 (OpenCV)
```

### Flux de Données : Du Jeu de Données au Modèle

```
[HuggingFace Hub]
        │
        ▼ datasets.load_dataset('kuzand/foodseg103')
[data/raw/foodseg103/]
   4983 images (.arrow format)
        │
        ▼ notebooks/02_preprocessing.ipynb
        │ - Analyse fréquence classes
        │ - Sélection top 12 classes
        │ - Split stratifié 70/15/15
        │ - Conversion masques → polygones
        │
[data/processed/]
   ├─ images/ (4526 JPEG 640×640)
   ├─ labels/ (4115 .txt polygones)
   └─ dataset.yaml (config YOLO)
        │
        ▼ notebooks/03_baseline_yolov8_medium.ipynb
        │ - Chargement yolov8m-seg.pt (COCO)
        │ - Fine-tuning 200 époques
        │ - Augmentations: mosaic, mixup, copy-paste
        │
[models/yolov8m_foodseg103/]
   └─ weights/best.pt (mAP50=0.617)
        │
        ▼ notebooks/04_quantization_export.ipynb
        │ - Export ONNX FP16
        │
[models/yolov8m_foodseg103/weights/best.onnx]
   (Prêt pour déploiement production)
```

---

### Architecture Application Web (Edge AI)

L'application NutriScan déployée fonctionne **entièrement dans le navigateur** (Edge AI). Il n'y a aucun serveur backend, aucune API REST. Le modèle ONNX est exécuté côté client via ONNX Runtime Web (WebAssembly) dans un Web Worker dédié.

**Application en ligne** : [https://app-computer-vision.vercel.app/](https://app-computer-vision.vercel.app/)

#### Architecture Client-Side

```
┌────────────────────────────────────────────────────────────────┐
│  Navigateur (client uniquement)                                │
│                                                                │
│  ┌────────────────────────────┐                                │
│  │  Main Thread (React)       │                                │
│  │  ├─ CameraScanner.tsx      │  ◄── Capture caméra            │
│  │  ├─ ImageUploader.tsx      │  ◄── Upload fichier            │
│  │  ├─ NutritionResult.tsx    │  ◄── Affichage résultats       │
│  │  └─ useInference.ts       │  ◄── Hook cycle de vie Worker   │
│  └───────────┬────────────────┘                                │
│              │ postMessage (image base64)                      │
│              ▼                                                 │
│  ┌────────────────────────────┐                                │
│  │  Web Worker                │                                │
│  │  (inference.worker.ts)     │                                │
│  │                            │                                │
│  │  1. Preprocessing          │  Canvas → Tensor [1,3,640,640] │
│  │  2. ONNX Runtime WASM      │  best.onnx (52 MB)             │
│  │  3. NMS (per + cross-class)│  Conf>0.25, IoU>0.45           │
│  │  4. Mask Generation        │  Coeffs × Protos → Sigmoid     │
│  │  5. Calcul Nutrition       │  Pixels → cm² → g → kcal       │
│  └───────────┬────────────────┘                                │
│              │ postMessage (détections + nutrition)            │
│              ▼                                                 │
│  Résultat: card par aliment + totaux kcal                      │
└────────────────────────────────────────────────────────────────┘
```

#### Structure des Modules TypeScript

```
app/src/
├── workers/
│   └── inference.worker.ts          # Web Worker : chargement ONNX + inférence
│
├── hooks/
│   ├── useInference.ts              # Hook React : init/run/cleanup Worker
│   └── useCamera.ts                 # Hook React : accès caméra getUserMedia
│
├── lib/
│   ├── inference/
│   │   ├── types.ts                 # Interfaces : Detection, NutritionInfo, etc.
│   │   ├── foodDatabase.ts          # Densité, épaisseur, kcal/100g (12 classes)
│   │   ├── preprocessing.ts         # Image → Float32Array NCHW [1,3,640,640]
│   │   ├── nms.ts                   # NMS per-class + cross-class confusion groups
│   │   └── postprocessing.ts        # Masques binaires + calcul nutritionnel
│   ├── workerClient.ts              # Communication Promise-based (message IDs)
│   └── constants.ts                 # Seuils, chemins modèle, calibration
│
├── components/
│   ├── CameraScanner.tsx            # Capture caméra temps réel
│   ├── ImageUploader.tsx            # Upload avec preview
│   ├── NutritionResult.tsx          # Résultats : totaux + détail par aliment
│   └── NutrientBar.tsx              # Barre de progression nutriment
│
└── app/page.tsx                     # Page principale (orchestration)
```

#### Communication Worker ↔ Main Thread

Le Worker utilise un protocole de messages avec IDs uniques :

```
Main Thread                          Web Worker
    │                                    │
    │── {id, type: "INIT"} ────────────► │ Charge best.onnx via ONNX Runtime
    │                                    │
    │◄── {id, type: "SUCCESS"} ──────────│
    │                                    │
    │── {id, type: "INFER",              │
    │    payload: base64} ─────────────► │ Preprocess → Infer → NMS → Masks
    │                                    │
    │◄── {id, type: "SUCCESS",           │
    │     payload: InferenceResult} ─────│
```

#### Déploiement Vercel

- **Hébergement** : Vercel (Next.js static)
- **Modèle ONNX** : servi depuis `/public/models/best.onnx` (52 MB, statique)
- **WASM** : fichiers ONNX Runtime chargés depuis CDN
- **Déploiement** : automatique à chaque push sur `main`

---
