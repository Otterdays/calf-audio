#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [[ ! -x codec/target/release/calf-codec ]]; then
  echo "Building calf-codec (Rust release)..."
  (cd codec && cargo build --release)
fi

exec uv run python app/desktop.py
