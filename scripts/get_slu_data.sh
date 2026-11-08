#!/usr/bin/env bash
# Download MixATIS / MixSNIPS (cleaned, Qin et al. 2020) from the official AGIF repository
# and verify them against the SHA-256 hashes fixed in docs/PREREGISTRATION_v2_confirmatory.md.
# The data is NOT redistributed in this repository: the AGIF repo is GPL-2.0 and the
# underlying ATIS corpus is distributed by the LDC under licence.
set -euo pipefail
cd "$(dirname "$0")/.."
base=https://raw.githubusercontent.com/LooperXX/AGIF/master/data
declare -A SHA=(
  [MixATIS_clean/train.txt]=bc54ad5e842975dce9a62e06303ea6d9e89754f74c3e56c8ae9b6e5257d99012
  [MixATIS_clean/dev.txt]=c07df0e5687d1445ccebf7e1f23bf8f63ac0d9b0fce7a51592354981dcf60d3f
  [MixATIS_clean/test.txt]=d68fdccdc2487e740c80f76164956604d594a4e86f8d492b434b64b407658f89
  [MixSNIPS_clean/train.txt]=bd05f8c6013a5df853f821124acafd0cf5eda272ddbb904ed075111efe51a4bc
  [MixSNIPS_clean/dev.txt]=cadd26a93d5c19dde8e35aca9d37c68a343f628ffbe0515a8e6a47f34bc6c7a9
  [MixSNIPS_clean/test.txt]=dcc6eaff01d939ac749745b0b49066e675159e571ebe7296d70a615d89d7ea0f
)
for f in "${!SHA[@]}"; do
  mkdir -p "data/external/$(dirname "$f")"
  [ -f "data/external/$f" ] || curl -sfL -o "data/external/$f" "$base/$f"
  got=$(sha256sum "data/external/$f" | cut -d' ' -f1)
  if [ "$got" != "${SHA[$f]}" ]; then echo "HASH MISMATCH: $f" >&2; exit 1; fi
  echo "ok  $f"
done
