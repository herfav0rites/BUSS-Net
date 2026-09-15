#!/usr/bin/env python3
"""Download and optionally extract a user-specified dataset archive."""

from __future__ import annotations

import argparse
import shutil
import urllib.request
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a dataset archive. An explicit URL is required to preserve provenance."
    )
    parser.add_argument("--url", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--extract-to", type=Path, default=None)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(args.url) as response, args.out.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    print(f"Downloaded {args.out} ({args.out.stat().st_size:,} bytes)")
    if args.extract_to is not None:
        args.extract_to.mkdir(parents=True, exist_ok=True)
        shutil.unpack_archive(str(args.out), str(args.extract_to))
        print(f"Extracted to {args.extract_to}")


if __name__ == "__main__":
    main()
