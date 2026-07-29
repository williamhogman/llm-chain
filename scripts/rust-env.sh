#!/usr/bin/env bash
# Shared environment for building the workspace in the Lovable sandbox.
# - Puts the Rust toolchain on PATH
# - Locates libclang (needed by bindgen when building llama-cpp-sys-2)
# - Points bindgen at the clang resource headers and glibc headers, which the
#   Nix-provided libclang does not find on its own
export PATH="$HOME/.cargo/bin:$PATH"
if [ -z "${LIBCLANG_PATH:-}" ]; then
  _libclang="$(find /nix/store -maxdepth 4 -name 'libclang.so*' 2>/dev/null | head -1)"
  if [ -n "$_libclang" ]; then
    export LIBCLANG_PATH="$(dirname "$_libclang")"
  fi
fi
if [ -z "${BINDGEN_EXTRA_CLANG_ARGS:-}" ] && [ -n "${LIBCLANG_PATH:-}" ]; then
  _clang_inc="$(find "$(dirname "$LIBCLANG_PATH")/lib/clang" -maxdepth 2 -name include 2>/dev/null | head -1)"
  _glibc_inc="$(find /nix/store -maxdepth 1 -name '*glibc*-dev' 2>/dev/null | head -1)/include"
  _args=""
  [ -n "$_clang_inc" ] && _args="-isystem $_clang_inc"
  [ -d "$_glibc_inc" ] && _args="$_args -isystem $_glibc_inc"
  if [ -n "$_args" ]; then
    export BINDGEN_EXTRA_CLANG_ARGS="$_args"
  fi
fi
