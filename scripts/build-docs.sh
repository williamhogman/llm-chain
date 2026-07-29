#!/usr/bin/env bash
# Builds the workspace rustdoc and stages it into dist/ so the docs can be
# published as a static site.
#
# The production/static-preview builder may not have a Rust toolchain. When
# cargo is available we build fresh docs AND refresh the committed snapshot
# (docs/site.tar.gz); when it is not, we unpack that snapshot so the build
# still produces a complete dist/.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SNAPSHOT="$ROOT_DIR/docs/site.tar.gz"

source "$SCRIPT_DIR/rust-env.sh"

if command -v cargo >/dev/null 2>&1; then
  cargo doc --workspace --no-deps

  rm -rf "$ROOT_DIR/dist"
  mkdir -p "$ROOT_DIR/dist"
  cp -r "$ROOT_DIR/target/doc/." "$ROOT_DIR/dist/"
  # cargo doc does not emit a root index.html; use the workspace landing page.
  cp "$SCRIPT_DIR/doc-index.html" "$ROOT_DIR/dist/index.html"

  # Refresh the committed snapshot so toolchain-less builders stay current.
  mkdir -p "$(dirname "$SNAPSHOT")"
  tar czf "$SNAPSHOT" -C "$ROOT_DIR/dist" .
else
  if [ ! -f "$SNAPSHOT" ]; then
    echo "error: cargo is unavailable and no docs snapshot exists at $SNAPSHOT" >&2
    exit 1
  fi
  echo "cargo not found; unpacking committed docs snapshot" >&2
  rm -rf "$ROOT_DIR/dist"
  mkdir -p "$ROOT_DIR/dist"
  tar xzf "$SNAPSHOT" -C "$ROOT_DIR/dist"
fi
