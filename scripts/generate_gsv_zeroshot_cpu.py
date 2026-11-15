from __future__ import annotations

import os
import sys
import time
import argparse
import io
import json
import math
from pathlib import Path


ROOT = Path(os.environ.get("VOCA_ROOT", Path(__file__).resolve().parent.parent))
REPO = ROOT / "frameworks/GPT-SoVITS"
sys.path.insert(0, str(ROOT))

from voca.audio import DEFAULT_OUTPUT_GAIN, DEFAULT_PEAK_LIMIT_DBFS, apply_gain_array, gain_db as multiplier_to_db

PROMPT_TEXT = "请按参考音色自然朗读。"
TARGET_RMS_DBFS = -27.0
TEXT = "这里填写要合成的台词。"
GSV_ITEM_PARAMETERS = {"top_k", "top_p", "temperature", "speed", "target_rms_dbfs", "output_gain", "peak_limit_dbfs"}


def load_jobs(batch_file: Path, output_dir: Path) -> list[dict[str, object]]:
    payload = json.loads(batch_file.read_text(encoding="utf-8"))
    entries = payload.get("items") if isinstance(payload, dict) else payload
    if not isinstance(entries, list) or not entries:
        raise ValueError("GSV batch file must contain a non-empty items array")
    jobs = []
    seen_outputs = set()
    for index, entry in enumerate(entries, 1):
        if not isinstance(entry, dict) or not isinstance(entry.get("text"), str) or not entry["text"].strip():
            raise ValueError(f"GSV batch item {index} needs non-empty text")
        name = str(entry.get("output", f"line-{index:03}.wav"))
        if Path(name).name != name or Path(name).suffix.lower() != ".wav":
            raise ValueError(f"GSV batch item {index} output must be a WAV filename without directories")
        if name in seen_outputs:
            raise ValueError(f"GSV batch output filename is repeated: {name}")
        seen_outputs.add(name)
        parameters = entry.get("parameters", {})
        if not isinstance(parameters, dict):
            raise ValueError(f"GSV batch item {index} parameters must be an object")
        unknown = sorted(set(parameters) - GSV_ITEM_PARAMETERS)
        if unknown:
            raise ValueError(f"GSV batch item {index} has unsupported parameter(s): {', '.join(unknown)}")
        for key, value in parameters.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"GSV batch item {index} parameter {key} must be finite and numeric")
            if key == "top_k" and (int(value) != value or int(value) < 1):
                raise ValueError(f"GSV batch item {index} top_k must be a positive integer")
            if key == "top_p" and not 0 <= float(value) <= 1:
                raise ValueError(f"GSV batch item {index} top_p must be between 0 and 1")
            if key in {"temperature", "speed", "output_gain"} and float(value) <= 0:
                raise ValueError(f"GSV batch item {index} {key} must be greater than 0")
            if key == "peak_limit_dbfs" and not -24 <= float(value) <= 0:
                raise ValueError(f"GSV batch item {index} peak_limit_dbfs must be between -24 and 0")
        jobs.append({"text": entry["text"].strip(), "output": output_dir / name, "parameters": parameters})
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT, help="Project-voca root")
    parser.add_argument("--pretrained-root", type=Path, help="GPT-SoVITS shared pretrained-assets directory")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--prompt-text", default=PROMPT_TEXT)
    parser.add_argument("--prompt-text-file", type=Path, help="UTF-8 file containing the reference transcript")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--text-file", type=Path)
    parser.add_argument("--batch-file", type=Path, help="JSON file with multiple text items; loads the model once")
    parser.add_argument("--output-dir", type=Path, help="Output directory for --batch-file")
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
    sovits_weight = args.sovits_weight or pretrained / "v2Pro/s2Gv2ProPlus.pth"
    gpt_weight = args.gpt_weight or pretrained / "s1v3.ckpt"
    if args.batch_file:
        if args.text_file:
            parser.error("--batch-file and --text-file cannot be used together")
        if not args.output_dir:
            parser.error("--batch-file requires --output-dir")
        output_dir = args.output_dir.expanduser().resolve()
        jobs = load_jobs(args.batch_file.expanduser().resolve(), output_dir)
    else:
        text = args.text_file.read_text(encoding="utf-8-sig").strip() if args.text_file else TEXT
        if not text:
            raise ValueError("Synthesis text is empty")
        output = args.output.expanduser().resolve() if args.output else root / "outputs/试听试验/若叶睦_GSV_v2ProPlus_零样本_汉堡独白.wav"
        jobs = [{"text": text, "output": output, "parameters": {}}]
    prompt_text = args.prompt_text_file.read_text(encoding="utf-8-sig").strip() if args.prompt_text_file else args.prompt_text
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
    import numpy as np
    import soundfile as sf

    batch_started = time.perf_counter()
    for index, job in enumerate(jobs, 1):
        item_parameters = job["parameters"]
        synthesis_text = api.cut_text(str(job["text"]), "。！？")
        print(f"Item {index}/{len(jobs)}: {len(synthesis_text.splitlines())} sentence chunks", flush=True)
        started = time.perf_counter()
        chunks = api.get_tts_wav(
            str(reference), prompt_text, "zh", synthesis_text, "zh",
            top_k=int(item_parameters.get("top_k", args.top_k)),
            top_p=float(item_parameters.get("top_p", args.top_p)),
            temperature=float(item_parameters.get("temperature", args.temperature)),
            speed=float(item_parameters.get("speed", args.speed)),
        )
        wav_bytes = next(chunks)
        audio, sample_rate = sf.read(io.BytesIO(wav_bytes), dtype="float32")
        rms = float(np.sqrt(np.mean(np.square(audio))))
        if rms == 0:
            raise RuntimeError(f"GSV returned silent audio for item {index}")
        target_rms = float(item_parameters.get("target_rms_dbfs", args.target_rms_dbfs))
        normalization_db = target_rms - 20.0 * math.log10(rms)
        normalization_gain = 10 ** (normalization_db / 20.0)
        peak = float(np.max(np.abs(audio)))
        if peak * normalization_gain > 0.98:
            normalization_gain = 0.98 / peak
            normalization_db = 20.0 * math.log10(normalization_gain)
        audio = audio * normalization_gain
        output_gain = float(item_parameters.get("output_gain", args.output_gain))
        peak_limit = float(item_parameters.get("peak_limit_dbfs", args.peak_limit_dbfs))
        audio = apply_gain_array(audio, output_gain, peak_limit)
        output_path = job["output"]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), audio, sample_rate, format="WAV", subtype="PCM_16")
        print(f"Saved {output_path}; RMS adjustment {normalization_db:+.2f} dB; "
              f"output gain ×{output_gain:g} ({multiplier_to_db(output_gain):+.2f} dB); "
              f"generation {time.perf_counter() - started:.1f}s", flush=True)
    print(f"Batch generation took {time.perf_counter() - batch_started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
