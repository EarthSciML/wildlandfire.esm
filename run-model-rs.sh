#!/usr/bin/env bash
# run-model-rs.sh — build+run the minimal Rust single-model runner.
#
# Thin wrapper over `cargo run` in run-model-rs/. The only wrinkle it handles is
# OpenSSL: the earthsci-toolkit crate pulls in s2bindings-sys, whose vendored
# s2geometry C++ kernel #include's <openssl/bn.h> and links libcrypto. On macOS
# those headers/libs live under the Homebrew prefix, which the compiler and
# linker do not search by default, so we add them via CPATH / LIBRARY_PATH
# (clang honors both for every compile/link regardless of the CMake build).
#
# Usage: ./run-model-rs.sh [model.esm] [t_end]
#   (defaults: ../wildlandfire.esm, t_end=2.0)

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if command -v brew >/dev/null 2>&1; then
  PREFIX="$(brew --prefix openssl@3 2>/dev/null || brew --prefix openssl 2>/dev/null || true)"
  if [[ -n "$PREFIX" ]]; then
    export OPENSSL_ROOT_DIR="${OPENSSL_ROOT_DIR:-$PREFIX}"
    export CPATH="$PREFIX/include${CPATH:+:$CPATH}"
    export LIBRARY_PATH="$PREFIX/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
  fi
fi

# --release: the array-expression interpreter is the hot path (see README
# "Profiling"); optimized it is ~4× faster than the debug build. The trade is a
# one-time ~4 min optimized compile of the toolkit; later runs reuse it.
exec cargo run --release --quiet --manifest-path "$HERE/run-model-rs/Cargo.toml" -- "$@"
