#!/usr/bin/env python3
"""Generate a Qwen3-TTS custom-voice source WAV on CPU."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--speaker", default="mujica_mutsumi")
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--text")
    parser.add_argument("--text-file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--output-gain", type=float, default=3.0,
                        help="Linear output gain multiplier; default is 3x")
    parser.add_argument("--peak-limit-dbfs", type=float, default=-1.0)
    args = parser.parse_args()
    if bool(args.text) == bool(args.text_file):
        parser.error("pass exactly one of --text or --text-file")

    root = args.root.expanduser().resolve()
    sys.path.insert(0, str(root))
    from voca.audio import apply_gain_array, gain_db

    model_path = args.model or root / "models/archive/qwen3-tts-bang-dream"
    text = args.text if args.text is not None else args.text_file.read_text(encoding="utf-8-sig").strip()
    if not text.strip():
        raise ValueError("Synthesis text is empty")
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel

    if torch.cuda.is_available():
        raise RuntimeError("CUDA is visible; refusing to run outside CPU mode")
    torch.set_num_threads(max(1, args.threads))
    torch.set_num_interop_threads(min(4, max(1, args.threads)))
    print(f"Loading Qwen model on CPU: {model_path}", flush=True)
    started = time.perf_counter()
    model = Qwen3TTSModel.from_pretrained(
        str(model_path), device_map="cpu", dtype=torch.float32, attn_implementation="sdpa"
    )
    print(f"Model loaded in {time.perf_counter() - started:.1f}s", flush=True)
    with torch.inference_mode():
        waveforms, sample_rate = model.generate_custom_voice(
            text=text, language=args.language, speaker=args.speaker
        )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    audio = apply_gain_array(waveforms[0], args.output_gain, args.peak_limit_dbfs)
    sf.write(str(output), audio, sample_rate, format="WAV", subtype="PCM_16")
    print(f"Saved {output}; {sample_rate} Hz; {len(audio) / sample_rate:.2f}s; "
          f"output gain ×{args.output_gain:g} ({gain_db(args.output_gain):+.2f} dB)", flush=True)


if __name__ == "__main__":
    main()
