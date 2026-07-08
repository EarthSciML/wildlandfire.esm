#!/usr/bin/env bash
# run-model-rs.sh — build+run the minimal Rust single-model runner.
#
# The Rust counterpart of run-model.jl / run-model-py.sh: runs the SAME
# terrain-coupled wildlandfire.esm, fetching the live USGS 3DEP elevation raster
# through the EarthSciIO Rust binding (its pure-Rust `geotiff` reader) and binding
# it to the model's USGS3DEP.raw.elevation loader field, then plots the fire front
# over that terrain. Requires a network connection (the terrain loader fetches
# live 3DEP; the raster is cached under run-model-rs/.esio-cache after the first
# run). The earthsci-ast + earthsciio crates are used straight from the
# sibling EarthSciAST / EarthSciIO checkouts by path — nothing vendored.
#
# Thin wrapper over `cargo run` in run-model-rs/. The one build wrinkle it handles
# is OpenSSL: the earthsci-ast crate pulls in s2bindings-sys, whose vendored
# s2geometry C++ kernel #include's <openssl/bn.h> and links libcrypto. On macOS
# those headers/libs live under the Homebrew prefix, which the compiler and
# linker do not search by default, so we add them via CPATH / LIBRARY_PATH
# (clang honors both for every compile/link regardless of the CMake build).
# (EarthSciIO's own HTTPS transport is rustls, so it needs no OpenSSL.)
#
# Usage: ./run-model-rs.sh [model.esm] [t_end]
#   (defaults: ../wildlandfire.esm, t_end=57600 = 16 h)

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
