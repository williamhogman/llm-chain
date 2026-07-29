#!/usr/bin/env bash
# Shared environment for building the workspace in the Lovable sandbox.
# - Puts the Rust toolchain on PATH (rustup shims or a toolchain bin dir)
# - Locates libclang (needed by bindgen when building llama-cpp-sys-2)
# - Points bindgen at the clang resource headers and glibc headers, which the
#   Nix-provided libclang does not find on its own
#
# This file is sourced from scripts that run under `set -euo pipefail`, so
# every probe below is written to never return a non-zero status.
export PATH="$HOME/.cargo/bin:/root/.cargo/bin:$PATH"

# Fall back to a rustup toolchain bin dir if the shims are missing.
if ! command -v cargo >/dev/null 2>&1; then
  for _tc in "$HOME"/.rustup/toolchains/*/bin /root/.rustup/toolchains/*/bin; do
    if [ -x "$_tc/cargo" ]; then
      export PATH="$_tc:$PATH"
      break
    fi
  done
fi

# llama-cpp-sys-2 builds llama.cpp with cmake; put it on PATH if needed.
if ! command -v cmake >/dev/null 2>&1; then
  _cmake_bin="$( (ls -d /nix/store/*cmake*/bin 2>/dev/null || true) | head -1)"
  if [ -n "$_cmake_bin" ] && [ -x "$_cmake_bin/cmake" ]; then
    export PATH="$_cmake_bin:$PATH"
  fi
fi


if [ -z "${LIBCLANG_PATH:-}" ]; then
  _libclang="$( (find /nix/store -maxdepth 4 -name 'libclang.so*' 2>/dev/null || true) | head -1)"
  if [ -n "$_libclang" ]; then
    export LIBCLANG_PATH="$(dirname "$_libclang")"
  fi
fi
if [ -z "${BINDGEN_EXTRA_CLANG_ARGS:-}" ] && [ -n "${LIBCLANG_PATH:-}" ]; then
  _clang_inc="$( (find "$(dirname "$LIBCLANG_PATH")/lib/clang" -maxdepth 2 -name include 2>/dev/null || true) | head -1)"
  _glibc_inc="$( (find /nix/store -maxdepth 1 -name '*glibc*-dev' 2>/dev/null || true) | head -1)/include"
  _args=""
  [ -n "$_clang_inc" ] && _args="-isystem $_clang_inc"
  [ -d "$_glibc_inc" ] && _args="$_args -isystem $_glibc_inc"
  if [ -n "$_args" ]; then
    export BINDGEN_EXTRA_CLANG_ARGS="$_args"
  fi
fi
