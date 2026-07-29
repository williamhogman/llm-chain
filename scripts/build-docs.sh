#!/usr/bin/env bash
# Builds the workspace rustdoc and stages it into dist/ so the docs can be
# published as a static site.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/rust-env.sh"

cargo doc --workspace --no-deps

rm -rf dist
mkdir -p dist
cp -r target/doc/. dist/
# cargo doc does not emit a root index.html; use the workspace landing page.
cp scripts/doc-index.html dist/index.html
