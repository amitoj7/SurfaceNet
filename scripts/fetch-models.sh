#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p android/app/libs models

download() {
  local url="$1" target="$2" digest="$3"
  if [[ -f "$target" ]] && printf '%s  %s\n' "$digest" "$target" | shasum -a 256 -c --status; then
    echo "$target verified"
    return
  fi
  curl --fail --location --retry 3 --continue-at - "$url" --output "$target.partial"
  printf '%s  %s\n' "$digest" "$target.partial" | shasum -a 256 -c
  mv "$target.partial" "$target"
}

download "https://ossci-android.s3.amazonaws.com/executorch/release/1.5.0-xnnpack/executorch.aar" \
  "android/app/libs/executorch.aar" "dcb50be130e1e45d898b846f30dedf2d627fc410191b3fb4e870a2b104b89286"
if [[ "${1:-}" != "--runtime-only" ]]; then
  base="https://huggingface.co/executorch-community/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8-ET/resolve/911221e0f49d3354cbd4697a9c09feee25cadec7"
  download "$base/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8.pte" "models/llama.pte" \
    "8715cdba9e91f6bede00cc5f2d6b12397b95225da5630c4972a8da03001cda3b"
  download "$base/tokenizer.model" "models/tokenizer.model" \
    "82e9d31979e92ab929cd544440f129d9ecd797b69e327f80f17e1c50d5551b55"
fi
