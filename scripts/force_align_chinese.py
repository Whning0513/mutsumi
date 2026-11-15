#!/usr/bin/env python3
"""Force-align a supplied Chinese transcript with MMS_FA and emit per-char times.

This helper is intended to run in an environment with torch, torchaudio, and
pypinyin installed. It deliberately does not perform speech recognition: the
transcript is treated as ground truth, romanized character by character, and
aligned to the acoustic emissions.
"""

import argparse
import json
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class Unit:
    index: int
    char: str
    pronunciation: str


def is_han(char: str) -> bool:
    code = ord(char)
    return (
        0x3400 <= code <= 0x4DBF
        or 0x4E00 <= code <= 0x9FFF
        or 0xF900 <= code <= 0xFAFF
        or 0x20000 <= code <= 0x3134F
    )


def normalize_pronunciation(value: str, allowed: set) -> str:
    value = unicodedata.normalize("NFKD", value.lower().replace("ü", "v"))
    value = "".join(char for char in value if not unicodedata.combining(char))
    value = value.replace("ü", "v")
    value = re.sub(r"[^a-z']", "", value)
    if not value or any(char not in allowed for char in value):
        raise ValueError("Unsupported pinyin syllable: {!r}".format(value))
    return value


def make_units(transcript: str, allowed: set, overrides: dict) -> List[Unit]:
    from pypinyin import Style, lazy_pinyin

    units: List[Unit] = []
    for char in transcript:
        if not char.isalnum():
            continue
        if char in overrides:
            pronunciation = normalize_pronunciation(str(overrides[char]), allowed)
        elif is_han(char):
            readings = lazy_pinyin(char, style=Style.NORMAL, strict=False)
            pronunciation = normalize_pronunciation(readings[0], allowed)
        elif char.isascii() and char.isalpha():
            # Latin text is aligned letter by letter. For acronyms or names
            # whose spoken form differs, replace them in the source with the
            # pronunciation used by the voice track.
            pronunciation = normalize_pronunciation(char, allowed)
        else:
            raise ValueError(
                "Cannot infer pronunciation for {!r}. Spell the spoken number "
                "out in Chinese characters before aligning.".format(char)
            )
        units.append(Unit(len(units), char, pronunciation))
    if not units:
        raise ValueError("Transcript contains no alignable Chinese or Latin characters.")
    return units


def score_spans(spans) -> float:
    frame_count = sum(span.end - span.start for span in spans)
    if frame_count <= 0:
        return 0.0
    return sum(span.score * (span.end - span.start) for span in spans) / frame_count


def find_speech_intervals(waveform, sample_rate: int):
    """Return coarse energy-based speech regions used to reject long CTC spills."""
    frame_length = max(1, round(sample_rate * 0.02))
    hop_length = max(1, round(sample_rate * 0.01))
    samples = waveform[0]
    if samples.numel() < frame_length:
        return []
    rms = samples.unfold(0, frame_length, hop_length).pow(2).mean(dim=1).sqrt()
    threshold = max(0.0005, float(rms.max()) * 0.008)
    active = (rms >= threshold).tolist()

    # Bridge only brief unvoiced gaps; longer pauses remain separate regions.
    max_gap = round(0.12 / (hop_length / sample_rate))
    index = 0
    while index < len(active):
        if active[index]:
            index += 1
            continue
        gap_start = index
        while index < len(active) and not active[index]:
            index += 1
        if gap_start > 0 and index < len(active) and index - gap_start <= max_gap:
            active[gap_start:index] = [True] * (index - gap_start)

    intervals = []
    start_index = None
    for index, is_active in enumerate(active + [False]):
        if is_active and start_index is None:
            start_index = index
        elif not is_active and start_index is not None:
            start = start_index * hop_length / sample_rate
            end = ((index - 1) * hop_length + frame_length) / sample_rate
            intervals.append([start, end])
            start_index = None
    return intervals


def trim_ctc_silence_spill(start: float, end: float, intervals) -> float:
    """Trim a single syllable if CTC carried it across a long silent gap."""
    if end - start <= 0.5:
        return end
    for left, right in zip(intervals, intervals[1:]):
        gap = right[0] - left[1]
        if gap >= 0.45 and start < left[1] and right[0] < end:
            return left[1]
    return end


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--transcript-file", required=True)
    parser.add_argument("--pinyin-map", help="Optional JSON map from individual characters to spoken pinyin")
    parser.add_argument("--output-json", default="-")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--threads", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch
    import torchaudio

    torch.set_num_threads(max(1, args.threads))
    bundle = torchaudio.pipelines.MMS_FA
    labels = bundle.get_dict()
    overrides = {}
    if args.pinyin_map:
        overrides = json.loads(Path(args.pinyin_map).read_text(encoding="utf-8"))
        if not isinstance(overrides, dict) or any(len(str(key)) != 1 for key in overrides):
            raise ValueError("Pinyin map must be a JSON object with single-character keys.")
    units = make_units(
        Path(args.transcript_file).read_text(encoding="utf-8-sig").strip(),
        set(labels),
        overrides,
    )

    waveform, sample_rate = torchaudio.load(args.audio)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != bundle.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)
        sample_rate = bundle.sample_rate

    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(Path(args.cache_dir)))
    device = torch.device("cpu")
    print(
        "Loading MMS_FA model on CPU (first run downloads about 1.3 GB)...",
        file=sys.stderr,
        flush=True,
    )
    model = bundle.get_model(with_star=False).to(device)
    tokenizer = bundle.get_tokenizer()
    aligner = bundle.get_aligner()
    targets = tokenizer([unit.pronunciation for unit in units])

    with torch.inference_mode():
        emissions, _ = model(waveform.to(device))
        token_spans = aligner(emissions[0], targets)

    frame_seconds = waveform.shape[1] / emissions.shape[1] / sample_rate
    activity = find_speech_intervals(waveform, sample_rate)
    aligned = []
    missing = []
    previous_start = -1.0
    for unit, spans in zip(units, token_spans):
        if not spans:
            missing.append(unit.char)
            continue
        raw_start = min(span.start for span in spans) * frame_seconds
        raw_end = max(span.end for span in spans) * frame_seconds
        if raw_end <= raw_start:
            missing.append(unit.char)
            continue
        start = raw_start
        end = trim_ctc_silence_spill(raw_start, raw_end, activity)
        if start + 1e-6 < previous_start:
            raise RuntimeError("Aligner returned non-monotonic character timestamps.")
        previous_start = start
        aligned.append(
            {
                "index": unit.index,
                "char": unit.char,
                "pronunciation": unit.pronunciation,
                "raw_start": raw_start,
                "raw_end": raw_end,
                "start": start,
                "end": end,
                "trimmed_silence": end < raw_end,
                "confidence": score_spans(spans),
            }
        )
    if missing:
        raise RuntimeError(
            "No acoustic span was assigned to {} transcript character(s): {}. "
            "Check that the supplied text matches the spoken audio.".format(
                len(missing), "".join(missing[:30])
            )
        )

    duration = waveform.shape[1] / sample_rate
    if any(item["end"] > duration + 0.15 for item in aligned):
        raise RuntimeError("Alignment extends beyond the audio duration.")
    result = {
        "model": "torchaudio.pipelines.MMS_FA",
        "device": "cpu",
        "pronunciation_overrides": overrides,
        "sample_rate": sample_rate,
        "audio_duration": duration,
        "mean_confidence": sum(item["confidence"] for item in aligned) / len(aligned),
        "speech_intervals": activity,
        "units": aligned,
    }
    rendered = json.dumps(result, ensure_ascii=False)
    if args.output_json == "-":
        print(rendered)
    else:
        Path(args.output_json).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ImportError, ValueError, RuntimeError) as error:
        print("error: {}".format(error), file=sys.stderr)
        sys.exit(2)
