# Documentation NutriScan (`app_docs`)

Site de documentation officiel du projet **NutriScan**, construit avec **Docusaurus + TypeScript**.

## Objectif

Cette application contient la documentation technique à jour du projet de vision par ordinateur pour l’estimation calorique à partir d’images alimentaires.

La documentation couvre notamment :
- Vue d’ensemble du projet
- Analyse technique
- Architecture logicielle
- Installation et reproduction
- Résultats expérimentaux
- Limitations et travaux futurs
- API
- Galerie des modèles

## Stack

- Docusaurus (preset classic)
- TypeScript strict
- `pnpm`
- Support mathématique `remark-math` + `rehype-katex`
- Icônes `lucide-react`

## Commandes (Windows / PowerShell)

Depuis le dossier `app_docs/` :

```bash
pnpm install
pnpm start
```

- `pnpm start` lance le serveur local de documentation.

```bash
pnpm typecheck
pnpm build
```

- `pnpm typecheck` valide la configuration et les composants TS/TSX.
- `pnpm build` génère le site statique dans `app_docs/build/`.

## Structure principale

- `docs/` : contenu Markdown/MDX de la documentation
- `sidebars.ts` : ordre manuel de la sidebar
- `docusaurus.config.ts` : config du site, thème, navbar, footer, plugins
- `src/pages/index.tsx` : page d’accueil
- `src/components/Icon.tsx` : composant d’icônes Lucide pour MDX
- `static/img/models/` : images des modèles utilisées dans la documentation

## Navigation documentation

- Entrée unique de la documentation : `/docs` (page **Vue D’Ensemble**)
- Sidebar ordonnée pour suivre le parcours de lecture
- Navigation manuelle dans les fichiers Markdown supprimée (gérée par Docusaurus)

## Notes

- Le site est configuré en français (`i18n: fr`).
- Le thème est optimisé pour les modes clair/sombre.
- Les images de `models/` sont synchronisées dans `static/img/models/` pour affichage web.
