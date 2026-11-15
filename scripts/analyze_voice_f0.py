#!/usr/bin/env python3
"""Summarize voiced fundamental-frequency statistics for speech audio."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import parselmouth


def summarize(path: Path, floor: float, ceiling: float, step: float, threshold: float):
    sound = parselmouth.Sound(str(path))
    pitch = sound.to_pitch_ac(
        time_step=step,
        pitch_floor=floor,
        pitch_ceiling=ceiling,
        silence_threshold=0.03,
        voicing_threshold=threshold,
        very_accurate=True,
    )
    values = pitch.selected_array["frequency"]
    voiced = values[np.isfinite(values) & (values > 0)]
    if not len(voiced):
        return sound, None
    percentiles = np.percentile(voiced, [10, 25, 50, 75, 90])
    voiced_ratio = len(voiced) / len(values)
    return sound, (percentiles, voiced_ratio, len(voiced), len(values))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", nargs="+", type=Path)
    parser.add_argument("--reference", type=Path, help="target-voice reference for semitone-difference estimates")
    parser.add_argument("--floor", type=float, default=60.0)
    parser.add_argument("--ceiling", type=float, default=500.0)
    parser.add_argument("--step", type=float, default=0.01)
    parser.add_argument("--voicing-threshold", type=float, default=0.45)
    args = parser.parse_args()

    all_paths = list(dict.fromkeys([*args.audio, *([args.reference] if args.reference else [])]))
    results = {}
    for path in all_paths:
        sound, summary = summarize(path, args.floor, args.ceiling, args.step, args.voicing_threshold)
        if summary is None:
            print(f"{path}: no voiced frames detected (duration={sound.duration:.2f}s)")
            continue
        p, ratio, n_voiced, n_total = summary
        results[path] = p[2]
        print(
            f"{path}: duration={sound.duration:.2f}s; "
            f"F0 p10/p25/median/p75/p90={p[0]:.1f}/{p[1]:.1f}/{p[2]:.1f}/{p[3]:.1f}/{p[4]:.1f} Hz; "
            f"voiced={ratio:.1%} ({n_voiced}/{n_total})"
        )

    if args.reference in results:
        target = results[args.reference]
        for path in args.audio:
            if path not in results or path == args.reference:
                continue
            delta = 12.0 * math.log2(target / results[path])
            print(f"median shift {path} -> reference: {delta:+.1f} semitones (rough starting point)")


if __name__ == "__main__":
    main()
