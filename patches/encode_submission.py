#!/usr/bin/env python3
"""Encode a Parameter Golf submission as a self-extracting lzma+base85 wrapper.

PR #1493's submission is wrapped this way so the human-readable train_gpt.py is
about 50KB but the on-disk file is ~16.6KB. The unwrapping is a one-line
exec(lzma.decompress(base64.b85decode(...))).

Usage:
  python3 encode_submission.py --in train_gpt_cwttt.py --out train_gpt.py
  python3 encode_submission.py --in <decoded.py> --out <submission.py> [--check]

The --check flag re-decodes after encoding and confirms byte-for-byte match.
"""

from __future__ import annotations

import argparse
import base64
import lzma
import sys
from pathlib import Path


FILTERS = [{"id": lzma.FILTER_LZMA2}]
WRAPPER = (
    "import lzma as L,base64 as B\n"
    'exec(L.decompress(B.b85decode("{blob}"),format=L.FORMAT_RAW,filters=[{{"id":L.FILTER_LZMA2}}]))'
)


def encode(src_bytes: bytes) -> str:
    compressed = lzma.compress(src_bytes, format=lzma.FORMAT_RAW, filters=FILTERS)
    blob = base64.b85encode(compressed).decode("ascii")
    return WRAPPER.format(blob=blob)


def decode(wrapped: str) -> bytes:
    import re

    m = re.search(r'b85decode\(\"(.+?)\"\)', wrapped, re.DOTALL)
    if not m:
        raise ValueError("could not find encoded blob in wrapper")
    return lzma.decompress(
        base64.b85decode(m.group(1)),
        format=lzma.FORMAT_RAW,
        filters=FILTERS,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True, type=Path)
    ap.add_argument("--out", dest="dst", required=True, type=Path)
    ap.add_argument("--check", action="store_true", help="round-trip verify after encode")
    args = ap.parse_args()

    src_bytes = args.src.read_bytes()
    print(f"[encode] source: {args.src} ({len(src_bytes)} bytes)")
    wrapped = encode(src_bytes)
    args.dst.write_text(wrapped)
    print(f"[encode] wrote:  {args.dst} ({len(wrapped)} bytes)")
    print(f"[encode] code-size budget: PR #1493 was 16,594 bytes; this is {len(wrapped)} (delta {len(wrapped) - 16594:+d})")

    if args.check:
        roundtrip = decode(wrapped)
        if roundtrip == src_bytes:
            print("[encode] roundtrip OK")
        else:
            print("[encode] FAIL: roundtrip differs from source")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
