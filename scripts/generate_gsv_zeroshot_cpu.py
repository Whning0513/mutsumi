from __future__ import annotations

import os
import sys
import time
import argparse
import io
import math
from pathlib import Path


ROOT = Path(os.environ.get("VOCA_ROOT", Path(__file__).resolve().parent.parent))
REPO = ROOT / "frameworks/GPT-SoVITS"
sys.path.insert(0, str(ROOT))

from voca.audio import DEFAULT_OUTPUT_GAIN, DEFAULT_PEAK_LIMIT_DBFS, apply_gain_array, gain_db as multiplier_to_db

PROMPT_TEXT = "请按参考音色自然朗读。"
TARGET_RMS_DBFS = -27.0
TEXT = "这里填写要合成的台词。"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT, help="Project-voca root")
    parser.add_argument("--pretrained-root", type=Path, help="GPT-SoVITS shared pretrained-assets directory")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--prompt-text", default=PROMPT_TEXT)
    parser.add_argument("--prompt-text-file", type=Path, help="UTF-8 file containing the reference transcript")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--text-file", type=Path)
    parser.add_argument("--sovits-weight", type=Path)
    parser.add_argument("--gpt-weight", type=Path)
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--top-p", type=float, default=0.6)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--target-rms-dbfs", type=float, default=TARGET_RMS_DBFS)
    parser.add_argument("--output-gain", type=float, default=DEFAULT_OUTPUT_GAIN,
                        help="Linear output gain multiplier; default is 3x")
    parser.add_argument("--peak-limit-dbfs", type=float, default=DEFAULT_PEAK_LIMIT_DBFS)
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    repo = root / "frameworks/GPT-SoVITS"
    pretrained = args.pretrained_root or root / "models/gpt-sovits/pretrained_models"
    reference = args.reference or root / "outputs/若叶睦_你好一二三为什么.wav"
    output = args.output or root / "outputs/试听试验/若叶睦_GSV_v2ProPlus_零样本_汉堡独白.wav"
    sovits_weight = args.sovits_weight or pretrained / "v2Pro/s2Gv2ProPlus.pth"
    gpt_weight = args.gpt_weight or pretrained / "s1v3.ckpt"
    text = args.text_file.read_text(encoding="utf-8-sig").strip() if args.text_file else TEXT
    prompt_text = args.prompt_text_file.read_text(encoding="utf-8-sig").strip() if args.prompt_text_file else args.prompt_text
    if not text:
        raise ValueError("Synthesis text is empty")
    if not reference.is_file():
        raise FileNotFoundError(reference)
    if not repo.is_dir():
        raise FileNotFoundError(repo)
    if not sovits_weight.is_file():
        raise FileNotFoundError(sovits_weight)
    if not gpt_weight.is_file():
        raise FileNotFoundError(gpt_weight)

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["version"] = "v2ProPlus"
    os.environ["is_half"] = "False"
    os.environ["bert_path"] = str(pretrained / "chinese-roberta-wwm-ext-large")
    os.environ["OMP_NUM_THREADS"] = "32"
    os.environ["MKL_NUM_THREADS"] = "32"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    os.chdir(repo)
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "GPT_SoVITS"))
    sys.argv = [
        "api.py",
        "-s",
        str(sovits_weight),
        "-g",
        str(gpt_weight),
        "-dr",
        str(reference),
        "-dt",
        prompt_text,
        "-dl",
        "zh",
        "-d",
        "cpu",
        "-fp",
        "-hb",
        str(pretrained / "chinese-hubert-base"),
        "-b",
        str(pretrained / "chinese-roberta-wwm-ext-large"),
    ]

    import torch

    if torch.cuda.is_available():
        raise RuntimeError("CPU-only run requested, but CUDA is visible")
    torch.set_num_threads(32)
    torch.set_num_interop_threads(4)

    if not prompt_text:
        raise ValueError("Reference prompt text is empty")
    print(f"Reference: {reference}", flush=True)
    print(f"SoVITS weight: {sovits_weight}", flush=True)
    print(f"GPT weight: {gpt_weight}", flush=True)
    print("Loading GPT-SoVITS v2 Pro Plus weights on CPU", flush=True)
    started = time.perf_counter()
    import api

    print(f"Models loaded in {time.perf_counter() - started:.1f}s", flush=True)
    synthesis_text = api.cut_text(text, "。！？")
    print(f"Sentence chunks: {len(synthesis_text.splitlines())}", flush=True)
    started = time.perf_counter()
    chunks = api.get_tts_wav(
        str(reference),
        prompt_text,
        "zh",
        synthesis_text,
        "zh",
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        speed=args.speed,
    )
    wav_bytes = next(chunks)
    import numpy as np
    import soundfile as sf

    audio, sample_rate = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    rms = float(np.sqrt(np.mean(np.square(audio))))
    if rms == 0:
        raise RuntimeError("GSV returned silent audio")
    gain_db = args.target_rms_dbfs - 20.0 * math.log10(rms)
    gain = 10 ** (gain_db / 20.0)
    peak = float(np.max(np.abs(audio)))
    if peak * gain > 0.98:
        gain = 0.98 / peak
        gain_db = 20.0 * math.log10(gain)
    audio = audio * gain
    audio = apply_gain_array(audio, args.output_gain, args.peak_limit_dbfs)
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output), audio, sample_rate, format="WAV", subtype="PCM_16")
    print(f"Saved {output}; RMS target adjustment {gain_db:+.2f} dB; output gain ×{args.output_gain:g} "
          f"({multiplier_to_db(args.output_gain):+.2f} dB), peak limit {args.peak_limit_dbfs:g} dBFS", flush=True)
    print(f"Generation took {time.perf_counter() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
