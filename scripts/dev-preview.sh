#!/usr/bin/env bash
# Serves the workspace rustdoc on port 8080 for the live preview.
# Starts the HTTP server immediately and (re)builds docs in the background.
set -u
export PATH="$HOME/.cargo/bin:$PATH"

mkdir -p target/doc

if [ ! -f target/doc/index.html ]; then
  cat > target/doc/index.html <<'EOF'
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>llm-chain — workspace docs</title>
<style>
  :root { color-scheme: dark; }
  body { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; background:#0f1115; color:#e6e6e6; margin:0; display:grid; place-items:center; min-height:100vh; }
  main { max-width: 640px; padding: 2rem; }
  h1 { font-size: 1.4rem; } h1 span { color:#f74c00; }
  a { color:#7aa2f7; text-decoration:none; } a:hover { text-decoration:underline; }
  ul { line-height: 2; list-style: "🦀 "; }
  p.small { color:#8b8fa3; font-size:.85rem; }
</style>
</head>
<body>
<main>
  <h1><span>llm-chain</span> workspace documentation</h1>
  <p>API docs are built with <code>cargo doc</code>. If a link 404s, the build is still running — refresh in a moment.</p>
  <ul>
    <li><a href="/llm_chain/index.html">llm-chain</a> — core traits, chains, prompt templates</li>
    <li><a href="/llm_chain_openai/index.html">llm-chain-openai</a> — OpenAI chat executor</li>
    <li><a href="/llm_chain_anthropic/index.html">llm-chain-anthropic</a> — Anthropic Claude executor</li>
    <li><a href="/llm_chain_gemini/index.html">llm-chain-gemini</a> — Google Gemini executor</li>
    <li><a href="/llm_chain_llama/index.html">llm-chain-llama</a> — llama.cpp (GGUF) executor</li>
    <li><a href="/llm_chain_tools/index.html">llm-chain-tools</a> — tools for agents</li>
  </ul>
  <p class="small">This page is served by <code>scripts/dev-preview.sh</code> for the Lovable live preview.</p>
</main>
</body>
</html>
EOF
fi

# Rebuild docs in the background; the placeholder index.html is never overwritten
# because cargo doc does not emit a root index.html.
( cargo doc --workspace --no-deps >/tmp/cargo-doc.log 2>&1 || true ) &

exec python3 -m http.server 8080 --directory target/doc
