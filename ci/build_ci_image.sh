#!/usr/bin/env bash
# Build the checked-out fork image for both core and nightly CI lanes.
set -euo pipefail

CUDA_LABEL=${1:?CUDA label is required}
DOCKER_IMAGE=${2:?Local image tag is required}
: "${GITHUB_ENV:?GitHub-compatible environment file is required}"

CUDA_CONFIG=$(jq -ce --arg label "${CUDA_LABEL}" '.runtime[] | select(.label == $label)' ci/cuda-versions.json)
docker build --target ci --file docker/Dockerfile.ci \
  --build-arg "CUDA_IMAGE=$(jq -r '.image' <<< "${CUDA_CONFIG}")" \
  --build-arg "PYTORCH_INDEX=$(jq -r '.pytorch_index' <<< "${CUDA_CONFIG}")" \
  --build-arg "CUDNN_VERSION=$(jq -r '.cudnn_version' <<< "${CUDA_CONFIG}")" \
  --tag "${DOCKER_IMAGE}" .
# Run exactly the image built from this checkout, never a stale Hub tag.
IMAGE_ID=$(docker image inspect --format '{{.Id}}' "${DOCKER_IMAGE}")
echo "DOCKER_IMAGE=${IMAGE_ID}" >> "$GITHUB_ENV"
docker run --rm --pull=never -i "${IMAGE_ID}" python3 - <<'PYTHON'
import pathlib, sys
expected = tuple(map(int, pathlib.Path('/install/.python-version').read_text().strip().split('.')))
assert sys.version_info[:len(expected)] == expected, (sys.version, expected)
print(sys.version)
PYTHON
