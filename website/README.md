# llm-chain website

The documentation website for `llm-chain`, published at
[docs.llm-chain.xyz](https://docs.llm-chain.xyz). Built with
[Docusaurus 3](https://docusaurus.io/).

## Development

```bash
cd website
npm install
npm start
```

Starts a local dev server at `http://localhost:3000` with hot reload.

## Build

```bash
npm run build
```

Generates the static site into `website/build`. The
`.github/workflows/website.yaml` workflow builds and deploys the site to
GitHub Pages on every push to `main`.
