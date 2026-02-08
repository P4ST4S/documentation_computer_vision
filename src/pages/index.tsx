import type {ReactNode} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import {
  BookOpenText,
  Brain,
  Database,
  FileText,
  Gauge,
  Image,
  PlayCircle,
  Rocket,
} from 'lucide-react';

import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <Heading as="h1" className="hero__title">
          {siteConfig.title}
        </Heading>
        <p className="hero__subtitle">{siteConfig.tagline}</p>
        <div className={styles.buttons}>
          <Link className="button button--secondary button--lg" to="/docs">
            <BookOpenText size={18} className={styles.buttonIcon} />
            Ouvrir La Documentation
          </Link>
          <Link className="button button--outline button--lg" to="/docs/model-gallery">
            <Image size={18} className={styles.buttonIcon} />
            Galerie Des Modèles
          </Link>
          <Link className="button button--secondary button--lg" href="https://app-computer-vision.vercel.app/">
            <Rocket size={18} className={styles.buttonIcon} />
            Tester l&apos;Application
          </Link>
        </div>
      </div>
    </header>
  );
}

export default function Home(): ReactNode {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`${siteConfig.title} Documentation`}
      description="Documentation technique Nutriscan pour la vision par ordinateur et l'estimation calorique.">
      <HomepageHeader />
      <main>
        <section className={styles.section}>
          <div className="container">
            <Heading as="h2" className={styles.sectionTitle}>
              <Gauge size={22} className={styles.titleIcon} />
              Résultats Clés
            </Heading>
            <div className={styles.metricsGrid}>
              <article className={styles.metricCard}>
                <Brain size={20} />
                <h3>Modèle</h3>
                <p>YOLOv8m-seg Fusion (32 classes)</p>
              </article>
              <article className={styles.metricCard}>
                <Gauge size={20} />
                <h3>Performance</h3>
                <p>mAP50 = 0.672, mAP50-95 = 0.565</p>
              </article>
              <article className={styles.metricCard}>
                <Database size={20} />
                <h3>Jeu De Données</h3>
                <p>FoodSeg103 + UEC-FoodPix (15 994 images)</p>
              </article>
              <article className={styles.metricCard}>
                <Rocket size={20} />
                <h3>Pile Technique</h3>
                <p>PyTorch, YOLOv8, OpenCV, MLflow</p>
              </article>
            </div>
          </div>
        </section>

        <section className={styles.section}>
          <div className="container">
            <Heading as="h2" className={styles.sectionTitle}>
              <Image size={22} className={styles.titleIcon} />
              Aperçu Visuel
            </Heading>
            <figure className={styles.figure}>
              <img
                src="/img/models/yolov8_fusion/val_batch0_pred.jpg"
                alt="Segmentation automatique d'aliments avec masques par instance (32 classes)"
                className={styles.preview}
              />
              <figcaption>
                Segmentation automatique d&apos;aliments avec masques par instance (YOLOv8m-seg Fusion, 32 classes).
              </figcaption>
            </figure>
          </div>
        </section>

        <section className={styles.section}>
          <div className="container">
            <Heading as="h2" className={styles.sectionTitle}>
              <FileText size={22} className={styles.titleIcon} />
              Documentation Complète
            </Heading>
            <p className={styles.lead}>
              La documentation technique complète est disponible dans la section Docs.
            </p>
            <div className={styles.docLinks}>
              <Link to="/docs">Vue d&apos;ensemble</Link>
              <Link to="/docs/analyse_technique">Analyse technique approfondie</Link>
              <Link to="/docs/architecture_logicielle">Architecture logicielle</Link>
              <Link to="/docs/installation">Installation et reproduction</Link>
              <Link to="/docs/resultats_experimentaux">Résultats Expérimentaux</Link>
              <Link to="/docs/limitations_futurs">Limitations Et Travaux Futurs</Link>
              <Link to="/docs/annexes">Annexes</Link>
            </div>
          </div>
        </section>

        <section className={styles.section}>
          <div className="container">
            <Heading as="h2" className={styles.sectionTitle}>
              <PlayCircle size={22} className={styles.titleIcon} />
              Démarrage Rapide
            </Heading>
            <pre className={styles.codeBlock}>
              <code>{`# Installation
python3 -m venv venv
source venv/bin/activate  # Windows: venv\\Scripts\\activate
pip install -r requirements.txt

# Préparation du dataset fusionné (32 classes)
jupyter notebook notebooks/02_data_fusion_and_cleaning.ipynb

# Entraînement du modèle Fusion
jupyter notebook notebooks/03_train_yolov8_fusion.ipynb

# Visualiser les métriques MLflow
mlflow ui`}</code>
            </pre>
          </div>
        </section>
      </main>
    </Layout>
  );
}
