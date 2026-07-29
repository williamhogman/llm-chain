#!/usr/bin/env bash
# Shared environment for building the workspace in the Lovable sandbox.
# - Puts the Rust toolchain on PATH
# - Locates libclang (needed by bindgen when building llama-cpp-sys-2)
export PATH="$HOME/.cargo/bin:$PATH"
if [ -z "${LIBCLANG_PATH:-}" ]; then
  _libclang="$(find /nix/store -maxdepth 4 -name 'libclang.so*' 2>/dev/null | head -1)"
  if [ -n "$_libclang" ]; then
    export LIBCLANG_PATH="$(dirname "$_libclang")"
  fi
fi
