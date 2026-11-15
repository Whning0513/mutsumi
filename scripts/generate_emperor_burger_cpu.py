#!/usr/bin/env python3
"""Generate the same private text with several Qwen custom-voice speakers."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path


DEFAULT_SPEAKERS = {
    "mutsumi": "mujica_mutsumi",
    "mortis": "mujica_mortis",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--text-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    model_path = args.model or root / "models/archive/qwen3-tts-bang-dream"
    output_dir = args.output_dir or root / "outputs"
    text = args.text_file.expanduser().read_text(encoding="utf-8-sig").strip()
    if not text:
        raise ValueError("Text file is empty")
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("OMP_NUM_THREADS", "32")
    os.environ.setdefault("MKL_NUM_THREADS", "32")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel

    if torch.cuda.is_available():
        raise RuntimeError("CUDA is visible; refusing to run outside CPU mode")
    torch.set_num_threads(32)
    torch.set_num_interop_threads(4)
    print(f"Loading model: {model_path}", flush=True)
    started = time.perf_counter()
    model = Qwen3TTSModel.from_pretrained(str(model_path), device_map="cpu", dtype=torch.float32,
                                           attn_implementation="sdpa")
    print(f"Model loaded in {time.perf_counter() - started:.1f}s", flush=True)
    for label, speaker in DEFAULT_SPEAKERS.items():
        output_path = output_dir / f"{label}.wav"
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists; pass --overwrite: {output_path}")
        started = time.perf_counter()
        with torch.inference_mode():
            waveforms, sample_rate = model.generate_custom_voice(text=text, language="Chinese", speaker=speaker)
        sf.write(str(output_path), waveforms[0], sample_rate, format="WAV", subtype="PCM_16")
        print(f"Saved {output_path}; {sample_rate} Hz; {len(waveforms[0]) / sample_rate:.2f}s; "
              f"inference {time.perf_counter() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
