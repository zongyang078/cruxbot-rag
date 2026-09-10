#!/usr/bin/env bash
# Download the demo index: 40,000 chunks, ~300 MB unpacked.
#
# The full index is 2.5 GB and is not published. This subset is a 10% sample
# that includes every passage the benchmark judged, so the queries in
# benchmarks/queries.jsonl return the passages behind the README's numbers.
set -euo pipefail

REPO="${CRUXBOT_REPO:-zongyang078/cruxbot-rag}"
TAG="${CRUXBOT_INDEX_TAG:-demo-index-v1}"
ASSET="cruxbot-demo-index.tar.gz"
URL="https://github.com/${REPO}/releases/download/${TAG}/${ASSET}"

cd "$(dirname "$0")/.."

if [ -d data/demo/chroma ]; then
  echo "data/demo already exists; delete it to re-download."
  exit 0
fi

mkdir -p data
echo "Fetching ${ASSET} from ${TAG} (~167 MB) ..."
curl -fL --progress-bar "$URL" -o "data/${ASSET}"

echo "Unpacking ..."
tar -xzf "data/${ASSET}" -C data
rm "data/${ASSET}"

echo "Ready: $(du -sh data/demo | cut -f1) in data/demo"
