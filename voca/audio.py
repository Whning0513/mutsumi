"""Audio gain helpers shared by synthesis entry points."""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_GAIN = 3.0
DEFAULT_PEAK_LIMIT_DBFS = -1.0


def gain_db(multiplier: float) -> float:
    if not math.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("Output gain must be a finite positive multiplier")
    return 20.0 * math.log10(multiplier)


def apply_gain_array(audio: Any, multiplier: float = DEFAULT_OUTPUT_GAIN,
                     peak_limit_dbfs: float = DEFAULT_PEAK_LIMIT_DBFS) -> Any:
    """Apply linear gain and a smooth peak limiter to a NumPy audio array."""
    import numpy as np

    gain_db(multiplier)
    if not math.isfinite(peak_limit_dbfs) or not -24.0 <= peak_limit_dbfs <= 0:
        raise ValueError("Peak limit must be between -24 and 0 dBFS")
    if multiplier == 1.0:
        return np.asarray(audio, dtype=np.float32)
    ceiling = 10.0 ** (peak_limit_dbfs / 20.0)
    knee = ceiling * 0.90
    gained = np.asarray(audio, dtype=np.float32) * float(multiplier)
    magnitude = np.abs(gained)
    above_knee = magnitude > knee
    limited_magnitude = magnitude.copy()
    width = ceiling - knee
    limited_magnitude[above_knee] = knee + width * np.tanh((magnitude[above_knee] - knee) / width)
    return np.sign(gained) * limited_magnitude


def apply_gain_to_file(path: str | Path, multiplier: float = DEFAULT_OUTPUT_GAIN,
                       peak_limit_dbfs: float = DEFAULT_PEAK_LIMIT_DBFS) -> None:
    """Apply gain to a synthesized audio file in place with FFmpeg's lookahead limiter."""
    audio_path = Path(path).expanduser().resolve()
    gain_db(multiplier)
    if not math.isfinite(peak_limit_dbfs) or not -24.0 <= peak_limit_dbfs <= 0:
        raise ValueError("Peak limit must be between -24 and 0 dBFS")
    if multiplier == 1.0:
        return
    if not audio_path.is_file():
        raise FileNotFoundError(audio_path)

    suffix = audio_path.suffix.lower()
    codec_args = {
        ".wav": ["-c:a", "pcm_s16le"],
        ".flac": ["-c:a", "flac"],
        ".mp3": ["-c:a", "libmp3lame", "-q:a", "2"],
        ".m4a": ["-c:a", "aac", "-b:a", "256k"],
        ".ogg": ["-c:a", "libvorbis", "-q:a", "5"],
    }.get(suffix)
    if codec_args is None:
        raise ValueError(f"Unsupported synthesis audio format for gain adjustment: {suffix or '(no extension)'}")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise FileNotFoundError("FFmpeg is required to apply the default output gain")

    ceiling = 10.0 ** (peak_limit_dbfs / 20.0)
    temporary = audio_path.with_name(f".{audio_path.stem}.gain-{uuid.uuid4().hex[:8]}{suffix}")
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(audio_path),
        "-map", "0:a:0", "-af", f"volume={multiplier:.8f},alimiter=limit={ceiling:.8f}:level=0",
        *codec_args, "-map_metadata", "0", str(temporary),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f"FFmpeg gain adjustment failed: {(result.stderr or '')[-2000:]}")
        os.replace(temporary, audio_path)
    finally:
        if temporary.exists():
            temporary.unlink()
