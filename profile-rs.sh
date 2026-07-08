#!/usr/bin/env bash
# profile-rs.sh — CPU-profile the Rust simulate() with pprof-rs.
#
# Builds+runs the `profile` bin under the `profiling` cargo profile (release
# optimization + debug symbols) with the `profile` feature (pulls in pprof-rs).
# Same OpenSSL env wrinkle as run-model-rs.sh (s2geometry needs <openssl/bn.h>).
# Outputs land in ./profiles/ (flamegraph-rust.svg, profile-rust-top.txt).
#
# Profiles the real three-loader simulate() (3DEP + LANDFIRE + ERA5, regridded
# in-model, at the runner's NX=18, NY=20); needs network + ~/.cdsapirc (ERA5 CDS).
# Usage: ./profile-rs.sh [t_end]     (default t_end=3600 s; the build-time regrid dominates)

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

[[ $# -ge 1 ]] && export PROFILE_TEND="$1"

exec cargo run --profile profiling --features profile --bin profile \
     --manifest-path "$HERE/run-model-rs/Cargo.toml"
