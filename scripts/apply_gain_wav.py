#!/usr/bin/env python3
"""Apply the project's default gain and peak limiter to an existing WAV."""

from __future__ import annotations

import argparse
import os
import sys
import uuid
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-gain", type=float, default=3.0)
    parser.add_argument("--peak-limit-dbfs", type=float, default=-1.0)
    args = parser.parse_args()
    path = args.input.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import soundfile as sf
    from voca.audio import apply_gain_array, gain_db

    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    adjusted = apply_gain_array(audio, args.output_gain, args.peak_limit_dbfs)
    temporary = path.with_name(f".{path.stem}.gain-{uuid.uuid4().hex[:8]}{path.suffix}")
    try:
        sf.write(str(temporary), adjusted, sample_rate, format="WAV", subtype="PCM_16")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(f"Applied ×{args.output_gain:g} ({gain_db(args.output_gain):+.2f} dB), "
          f"peak limit {args.peak_limit_dbfs:g} dBFS: {path}", flush=True)


if __name__ == "__main__":
    main()
