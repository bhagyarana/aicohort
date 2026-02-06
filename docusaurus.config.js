// @ts-check
const lightCodeTheme = require('prism-react-renderer/themes/github');
const darkCodeTheme = require('prism-react-renderer/themes/dracula');

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Agentic AI Training',
  tagline: 'Master the art of building intelligent AI agents',
  url: 'https://agentic-ai-training.com',
  baseUrl: '/',
  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',
  favicon: 'img/favicon.ico',

  organizationName: 'aicohort',
  projectName: 'agentic-ai-training',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          routeBasePath: 'learn',
        },
        blog: false,
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      colorMode: {
        defaultMode: 'light',
        disableSwitch: false,
        respectPrefersColorScheme: true,
      },
      navbar: {
        title: 'Agentic AI',
        logo: {
          alt: 'Agentic AI Training Logo',
          src: 'img/logo.svg',
        },
        items: [
          {
            type: 'doc',
            docId: 'onboarding/index',
            position: 'left',
            label: 'Get Started',
          },
          {
            type: 'doc',
            docId: 'modules/index',
            position: 'left',
            label: 'Modules',
          },
          {
            type: 'doc',
            docId: 'agent-patterns/index',
            position: 'left',
            label: 'Agent Patterns',
          },
          {
            type: 'doc',
            docId: 'capstone/index',
            position: 'left',
            label: 'Capstone',
          },
          {
            href: 'https://github.com/aicohort/agentic-ai-training',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Learning Paths',
            items: [
              {
                label: 'Get Started',
                to: '/learn/onboarding',
              },
              {
                label: 'Module 1: Foundations',
                to: '/learn/modules/module-1',
              },
              {
                label: 'Capstone Project',
                to: '/learn/capstone',
              },
            ],
          },
          {
            title: 'Resources',
            items: [
              {
                label: 'LangChain Docs',
                href: 'https://python.langchain.com/docs/',
              },
              {
                label: 'OpenAI API',
                href: 'https://platform.openai.com/docs',
              },
              {
                label: 'Google AI',
                href: 'https://ai.google.dev/',
              },
            ],
          },
          {
            title: 'Community',
            items: [
              {
                label: 'Discord',
                href: 'https://discord.gg/aicohort',
              },
              {
                label: 'GitHub',
                href: 'https://github.com/aicohort',
              },
            ],
          },
        ],
        copyright: `Copyright ${new Date().getFullYear()} AI Cohort. Built for learning Agentic AI.`,
      },
      prism: {
        theme: lightCodeTheme,
        darkTheme: darkCodeTheme,
        additionalLanguages: ['python', 'bash', 'json'],
      },
      tableOfContents: {
        minHeadingLevel: 2,
        maxHeadingLevel: 4,
      },
    }),
};

module.exports = config;
