// @ts-check
import { themes as prismThemes } from "prism-react-renderer";

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: "llm-chain",
  tagline: "The ultimate LLM toolbox for Rust",
  favicon: "img/favicon.ico",

  url: "https://docs.llm-chain.xyz",
  baseUrl: "/",

  organizationName: "sobelio",
  projectName: "llm-chain",
  trailingSlash: false,

  onBrokenLinks: "throw",

  markdown: {
    hooks: {
      onBrokenMarkdownLinks: "warn",
    },
  },


  i18n: {
    defaultLocale: "en",
    locales: ["en"],
  },

  presets: [
    [
      "classic",
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: "./sidebars.js",
          editUrl: "https://github.com/sobelio/llm-chain/tree/main/website/",
        },
        blog: {
          showReadingTime: true,
          editUrl: "https://github.com/sobelio/llm-chain/tree/main/website/",
          onInlineAuthors: "ignore",
          onUntruncatedBlogPosts: "ignore",
        },
        theme: {
          customCss: "./src/css/custom.css",
        },
        gtag: {
          trackingID: "G-Q8CJDJT9GX",
          anonymizeIP: true,
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      image: "img/llmchainsocial.png",
      navbar: {
        title: "llm-chain",
        logo: {
          alt: "llm-chain logo",
          src: "img/llmchain.svg",
        },
        items: [
          {
            type: "docSidebar",
            sidebarId: "sidebar",
            position: "left",
            label: "Documentation",
          },
          { to: "/blog", label: "Blog", position: "left" },
          {
            href: "https://docs.rs/llm-chain",
            label: "API Reference",
            position: "right",
          },
          {
            href: "https://github.com/sobelio/llm-chain",
            label: "GitHub",
            position: "right",
          },
        ],
      },
      footer: {
        style: "dark",
        links: [
          {
            title: "Docs",
            items: [
              {
                label: "Introduction",
                to: "/docs/introduction",
              },
              {
                label: "Getting started",
                to: "/docs/getting-started-tutorial",
              },
            ],
          },
          {
            title: "Community",
            items: [
              {
                label: "Discord",
                href: "https://discord.gg/kewN9Gtjt2",
              },
              {
                label: "GitHub",
                href: "https://github.com/sobelio/llm-chain",
              },
            ],
          },
          {
            title: "More",
            items: [
              {
                label: "Blog",
                to: "/blog",
              },
              {
                label: "Docs.rs",
                href: "https://docs.rs/llm-chain",
              },
              {
                label: "Crates.io",
                href: "https://crates.io/crates/llm-chain",
              },
              {
                label: "Sobel.io",
                href: "https://sobel.io",
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} Sobel.io AB. Built with Docusaurus.`,
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
        additionalLanguages: ["rust", "toml", "bash", "yaml"],
      },
    }),
};

export default config;
