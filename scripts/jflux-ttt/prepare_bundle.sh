#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=/kmh-nfs-ssd-us-mount/code/xtiange/jflux2
BUILD=$(mktemp -d "$HERE/build.XXXXXX")
SNAP="$BUILD/source"

test -f "$SRC/jflux2/analysis/capacity_ladder_v8.py"
mkdir -p "$SNAP"
while IFS= read -r rel; do
  test -n "$rel" || continue
  test -f "$SRC/$rel" || { echo "missing source: $SRC/$rel" >&2; exit 1; }
  mkdir -p "$SNAP/$(dirname "$rel")"
  cp -p "$SRC/$rel" "$SNAP/$rel"
done < "$HERE/source_files.txt"

# LPIPS carries a small pretrained AlexNet weight file not listed as Python source.
if test -d "$SRC/third_party/lpips_pkg/lpips/weights"; then
  cp -a "$SRC/third_party/lpips_pkg/lpips/weights" "$SNAP/third_party/lpips_pkg/lpips/"
fi

patch -d "$SNAP" -p1 < "$HERE/central2_zone_assets.patch"
find "$SNAP" -type f -print0 | sort -z | xargs -0 sha256sum > "$HERE/SOURCE_SHA256SUMS.absolute"
sed "s#${SNAP}/##" "$HERE/SOURCE_SHA256SUMS.absolute" > "$HERE/SOURCE_SHA256SUMS"
rm "$HERE/SOURCE_SHA256SUMS.absolute"
tar -C "$BUILD" -cf "$HERE/run_bundle.tar" source
sha256sum "$HERE/run_bundle.tar" > "$HERE/run_bundle.tar.sha256"
echo "bundle ready: $HERE/run_bundle.tar"
