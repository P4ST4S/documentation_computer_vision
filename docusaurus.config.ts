import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';
import {themes as prismThemes} from 'prism-react-renderer';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

const config: Config = {
  title: 'Nutriscan',
  tagline: "Documentation de vision par ordinateur pour l'estimation calorique à partir d'images alimentaires",
  favicon: 'img/favicon.ico',
  url: 'https://example.com',
  baseUrl: '/',
  organizationName: 'nutriscan',
  projectName: 'app_docs',
  onBrokenLinks: 'throw',
  onDuplicateRoutes: 'throw',
  markdown: {
    hooks: {
      onBrokenMarkdownLinks: 'warn',
    },
  },
  i18n: {
    defaultLocale: 'fr',
    locales: ['fr'],
  },
  stylesheets: [
    {
      href: 'https://cdn.jsdelivr.net/npm/katex@0.16.28/dist/katex.min.css',
      type: 'text/css',
      crossorigin: 'anonymous',
    },
  ],
  presets: [
    [
      'classic',
      {
        docs: {
          path: 'docs',
          routeBasePath: 'docs',
          sidebarPath: './sidebars.ts',
          remarkPlugins: [remarkMath],
          rehypePlugins: [rehypeKatex],
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],
  themeConfig: {
    image: 'img/nutriscan-social-card.png',
    navbar: {
      title: 'Nutriscan',
      logo: {
        alt: 'Logo Nutriscan',
        src: 'img/logo.svg',
      },
      items: [
        {
          to: '/docs',
          label: 'Documentation',
          position: 'left',
        },
        {
          to: '/docs/api',
          label: 'API',
          position: 'left',
        },
        {
          href: 'https://github.com/your-org/nutriscan',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Documentation',
          items: [
            {label: 'Documentation', to: '/docs'},
            {label: 'API', to: '/docs/api'},
          ],
        },
        {
          title: 'Projet',
          items: [
            {label: 'Dépôt', href: 'https://github.com/your-org/nutriscan'},
          ],
        },
      ],
      copyright: `Droits réservés © ${new Date().getFullYear()} Nutriscan.`,
    },
    colorMode: {
      defaultMode: 'light',
      disableSwitch: false,
      respectPrefersColorScheme: true,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
    },
  },
};

export default config;
