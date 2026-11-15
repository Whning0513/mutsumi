#!/usr/bin/env python3
"""Force-align a supplied Chinese transcript and burn synchronized captions in.

The supplied transcript is ground truth. A multilingual CTC forced aligner
places each transcript character on the audio; no ASR or fuzzy text matching is
used to decide either caption text or timestamps.
"""

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import wave
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:  # pragma: no cover - exercised by the CLI error path
    raise SystemExit(
        "Pillow is required to render outlined Chinese text. Install it with: "
        "python3 -m pip install --user -r scripts/requirements-subtitles.txt"
    ) from exc


PUNCTUATION = set("，。！？；：、,.!?;:")
SUBTITLE_STYLES = ("outline", "dark-rectangle", "rounded-box", "yellow-highlight", "shadow")
FONT_CANDIDATES = [
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "C:/Windows/Fonts/msyh.ttc",
]


@dataclass
class TimedChar:
    char: str
    start: float
    end: float
    confidence: float


@dataclass
class SubtitleChunk:
    text: str
    char_indices: List[int]


@dataclass
class TimedCue:
    text: str
    raw_start: float
    raw_end: float
    start: float
    end: float


def find_executable(name: str, candidates: Sequence[str]) -> str:
    found = shutil.which(name)
    if found:
        return found
    for candidate in candidates:
        if Path(candidate).is_file():
            return candidate
    raise FileNotFoundError(
        "Could not find {}. Install it or pass its path on the command line.".format(name)
    )


def find_font(explicit: Optional[str]) -> str:
    candidates = ([explicit] if explicit else []) + FONT_CANDIDATES
    for candidate in candidates:
        if candidate and Path(candidate).expanduser().is_file():
            return str(Path(candidate).expanduser())
    raise FileNotFoundError(
        "No Chinese font found. Pass a font file with --font PATH."
    )


def speech_char(char: str) -> bool:
    """Keep written speech characters and ignore punctuation and emoji."""
    return char.isalnum()


def run_forced_alignment(
    audio_path: Path,
    transcript_path: Path,
    pinyin_map_path: Optional[Path],
    helper_path: Path,
    remote_host: str,
    remote_python: str,
    remote_cache: str,
    threads: int,
) -> dict:
    """Run the CPU MMS_FA aligner over SSH and return its per-character JSON."""
    stage_id = "subtitle-align-{}-{}".format(os.getpid(), uuid.uuid4().hex[:8])
    remote_dir = "/tmp/{}".format(stage_id)
    remote_helper = remote_dir + "/force_align_chinese.py"
    remote_audio = remote_dir + "/audio" + audio_path.suffix.lower()
    remote_transcript = remote_dir + "/transcript.txt"
    remote_pinyin_map = remote_dir + "/pinyin-map.json"

    def run(command: Sequence[str], capture: bool = False) -> str:
        result = subprocess.run(
            list(command),
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.PIPE if capture else None,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode:
            detail = result.stderr[-5000:] if capture and result.stderr else ""
            raise RuntimeError(
                "Command failed ({}): {}{}".format(
                    result.returncode,
                    " ".join(command[:3]),
                    "\n" + detail if detail else "",
                )
            )
        return result.stdout or "" if capture else ""

    try:
        run(["ssh", "-T", remote_host, "mkdir -p {}".format(shlex.quote(remote_dir))])
        run([
            "scp", str(helper_path), "{}:{}".format(remote_host, remote_helper),
        ])
        run([
            "scp", str(audio_path), "{}:{}".format(remote_host, remote_audio),
        ])
        run([
            "scp", str(transcript_path), "{}:{}".format(remote_host, remote_transcript),
        ])
        if pinyin_map_path:
            run([
                "scp", str(pinyin_map_path), "{}:{}".format(remote_host, remote_pinyin_map),
            ])

        print("Running transcript forced alignment on {} (CPU)...".format(remote_host), flush=True)
        remote_command = " ".join(
            shlex.quote(part)
            for part in [
                remote_python,
                remote_helper,
                "--audio", remote_audio,
                "--transcript-file", remote_transcript,
                "--output-json", "-",
                "--cache-dir", remote_cache,
                "--threads", str(threads),
            ]
            + (["--pinyin-map", remote_pinyin_map] if pinyin_map_path else [])
        )
        result = subprocess.run(
            ["ssh", "-T", remote_host, remote_command],
            stdout=subprocess.PIPE,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode:
            raise RuntimeError(
                "Remote forced aligner failed ({}). Check that torch, torchaudio, "
                "and pypinyin are installed in {}.".format(result.returncode, remote_python)
            )
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Remote aligner returned invalid JSON: {}".format(exc)) from exc
    finally:
        subprocess.run(
            ["ssh", "-T", remote_host, "rm -rf {}".format(shlex.quote(remote_dir))],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def split_chunks(
    transcript: str,
    aligned: Sequence[Optional[TimedChar]],
    max_chars: int,
    chars_per_line: int,
    max_gap: float,
    max_duration: float,
) -> List[SubtitleChunk]:
    phrases: List[SubtitleChunk] = []
    current_text: List[str] = []
    current_indices: List[int] = []
    char_index = 0

    def flush() -> None:
        nonlocal current_text, current_indices
        text = "".join(current_text).strip()
        if text and current_indices:
            phrases.append(SubtitleChunk(text, list(current_indices)))
        current_text = []
        current_indices = []

    for char in transcript:
        if char in "\r\n":
            flush()
            continue
        if speech_char(char) and len(current_indices) >= max_chars:
            flush()
        current_text.append(char)
        if speech_char(char):
            current_indices.append(char_index)
            char_index += 1
        if char in PUNCTUATION:
            flush()
    flush()

    chunks: List[SubtitleChunk] = []
    line_texts: List[str] = []
    line_indices: List[List[int]] = []
    total_chars = 0

    def flush_card() -> None:
        nonlocal line_texts, line_indices, total_chars
        text = "\n".join(line_texts).strip()
        indices = [index for line in line_indices for index in line]
        if text and indices:
            chunks.append(SubtitleChunk(text, indices))
        line_texts = []
        line_indices = []
        total_chars = 0

    def phrase_times(phrase: SubtitleChunk) -> Tuple[float, float]:
        timed = [aligned[index] for index in phrase.char_indices if index < len(aligned)]
        timed = [item for item in timed if item is not None]
        if not timed:
            return 0.0, 0.0
        return timed[0].start, timed[-1].end

    current_start = 0.0
    current_end = 0.0
    for phrase in phrases:
        phrase_start, phrase_end = phrase_times(phrase)
        phrase_chars = len(phrase.char_indices)
        if line_texts:
            gap = phrase_start - current_end
            duration = phrase_end - current_start
            too_long = (
                total_chars + phrase_chars > max_chars
                or gap > max_gap
                or duration > max_duration
                or (
                    len(line_texts) >= 2
                    and len(line_indices[-1]) + phrase_chars > chars_per_line
                )
                or (len(line_texts) == 1 and len(line_indices[0]) > chars_per_line)
            )
            if too_long:
                flush_card()

        if not line_texts:
            line_texts = [phrase.text]
            line_indices = [list(phrase.char_indices)]
            total_chars = phrase_chars
            current_start = phrase_start
            current_end = phrase_end
            continue

        if len(line_indices[-1]) + phrase_chars > chars_per_line:
            line_texts.append(phrase.text)
            line_indices.append(list(phrase.char_indices))
        else:
            line_texts[-1] += phrase.text
            line_indices[-1].extend(phrase.char_indices)
        total_chars += phrase_chars
        current_end = phrase_end

    flush_card()
    return chunks


def chunk_bounds(chunk: SubtitleChunk, aligned: Sequence[Optional[TimedChar]]) -> Tuple[float, float, float]:
    timed = [aligned[index] for index in chunk.char_indices if index < len(aligned)]
    timed = [item for item in timed if item is not None]
    if not timed:
        raise ValueError("No forced-alignment times were found for subtitle: {}".format(chunk.text))

    # Repeated interjections such as 啊啊啊 are aligned character by character;
    # use the full run when computing the visible cue bounds.
    normalized = [char.lower() for char in chunk.text if speech_char(char)]
    repeated = len(normalized) > 1 and len(set(normalized)) == 1
    if repeated:
        start = min(item.start for item in timed)
        end = max(item.end for item in timed)
    else:
        start = timed[0].start
        end = timed[-1].end
        if end < start:
            start = min(item.start for item in timed)
            end = max(item.end for item in timed)
    confidence = sum(item.confidence for item in timed) / len(timed)
    return start, max(start + 0.04, end), confidence


def make_cues(
    chunks: Sequence[SubtitleChunk],
    aligned: Sequence[Optional[TimedChar]],
    duration: float,
    pad_before: float,
    pad_after: float,
) -> List[TimedCue]:
    cues: List[TimedCue] = []
    for chunk in chunks:
        raw_start, raw_end, confidence = chunk_bounds(chunk, aligned)
        if confidence < 0.03:
            raise ValueError(
                "Forced-alignment confidence is very low ({:.1%}) for: {}. Check the "
                "spoken pronunciation against the transcript.".format(confidence, chunk.text)
            )
        start = max(0.0, raw_start - pad_before)
        end = min(duration, raw_end + pad_after)
        if end <= start:
            raise ValueError("Invalid subtitle timing for: {}".format(chunk.text))
        cues.append(TimedCue(chunk.text, raw_start, raw_end, start, end))

    # Hold a comma-ended phrase through most of a short natural pause, but
    # switch to the next phrase before its first spoken character.
    for index in range(len(cues) - 1):
        current = cues[index]
        following = cues[index + 1]
        gap = following.raw_start - current.raw_end
        if current.text.rstrip().endswith(("，", ",")) and 0 < gap <= 0.5:
            hold = min(0.22, gap * 0.8)
            current.end = min(
                max(current.end, current.raw_end + hold),
                following.start - 0.02,
            )

    # Keep subtitle cards from overlapping when adjacent word estimates touch.
    for index in range(len(cues) - 1):
        current = cues[index]
        following = cues[index + 1]
        if current.end > following.start:
            boundary = (current.raw_end + following.raw_start) / 2.0
            boundary = max(current.start + 0.05, min(boundary, following.end - 0.05))
            current.end = boundary
            following.start = boundary
    return cues


def format_srt_time(seconds: float) -> str:
    milliseconds = max(0, int(round(seconds * 1000)))
    hours, milliseconds = divmod(milliseconds, 3600000)
    minutes, milliseconds = divmod(milliseconds, 60000)
    whole_seconds, milliseconds = divmod(milliseconds, 1000)
    return "{:02}:{:02}:{:02},{:03}".format(hours, minutes, whole_seconds, milliseconds)


def write_srt(cues: Sequence[TimedCue], path: Path) -> None:
    blocks = []
    for index, cue in enumerate(cues, 1):
        blocks.append(
            "{}\n{} --> {}\n{}".format(
                index, format_srt_time(cue.start), format_srt_time(cue.end), cue.text
            )
        )
    path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")


def wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int, stroke: int) -> List[str]:
    lines: List[str] = []
    line = ""
    for char in text:
        if char == "\n":
            lines.append(line)
            line = ""
            continue
        candidate = line + char
        box = font.getbbox(candidate, stroke_width=stroke)
        if line and box[2] - box[0] > max_width:
            lines.append(line)
            line = char
        else:
            line = candidate
    if line:
        lines.append(line)
    return lines or [text]


def render_caption(
    text: str,
    path: Path,
    width: int,
    height: int,
    font: ImageFont.FreeTypeFont,
    stroke: int,
    margin_bottom: int,
    style: str = "outline",
    box_opacity: int = 84,
) -> None:
    if style not in SUBTITLE_STYLES:
        raise ValueError("Unknown subtitle style: {}".format(style))
    if not 0 <= box_opacity <= 100:
        raise ValueError("Box opacity must be between 0 and 100")

    presets = {
        "outline": {"fill": (255, 255, 255, 255), "stroke": stroke, "stroke_fill": (0, 0, 0, 255)},
        "dark-rectangle": {
            "fill": (255, 255, 255, 255), "stroke": 0, "stroke_fill": (0, 0, 0, 0),
            "background": (42, 45, 49, round(255 * box_opacity / 100)), "padding": (30, 18),
        },
        "rounded-box": {
            "fill": (255, 255, 255, 255), "stroke": 0, "stroke_fill": (0, 0, 0, 0),
            "background": (18, 22, 29, round(255 * box_opacity / 100)), "padding": (38, 22), "radius": 24,
        },
        "yellow-highlight": {
            "fill": (255, 224, 78, 255), "stroke": max(2, round(stroke * 0.8)), "stroke_fill": (20, 20, 20, 255),
        },
        "shadow": {
            "fill": (255, 255, 255, 255), "stroke": 0, "stroke_fill": (0, 0, 0, 0),
            "shadow": (0, 0, 0, 185), "shadow_offset": (4, 5),
        },
    }
    preset = presets[style]
    text_stroke = int(preset["stroke"])
    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    padding_x, padding_y = preset.get("padding", (0, 0))
    lines = wrap_text(text, font, width - 96 - 2 * padding_x, text_stroke)
    if len(lines) > 2:
        raise ValueError("Subtitle needs more than two lines at the selected font size: {}".format(text))

    boxes = [draw.textbbox((0, 0), line, font=font, stroke_width=text_stroke) for line in lines]
    line_heights = [box[3] - box[1] for box in boxes]
    gap = max(2, round(font.size * 0.08))
    total_height = sum(line_heights) + gap * (len(lines) - 1)
    block_width = max(box[2] - box[0] for box in boxes)
    block_left = (width - block_width) / 2
    block_top = height - margin_bottom - total_height - 2 * padding_y
    if "background" in preset:
        box_left = block_left - padding_x
        box_top = block_top
        box_right = block_left + block_width + padding_x
        box_bottom = height - margin_bottom
        bounds = (round(box_left), round(box_top), round(box_right), round(box_bottom))
        if "radius" in preset:
            draw.rounded_rectangle(bounds, radius=int(preset["radius"]), fill=preset["background"])
        else:
            draw.rectangle(bounds, fill=preset["background"])
    y = block_top + padding_y
    for line, box, line_height in zip(lines, boxes, line_heights):
        line_width = box[2] - box[0]
        x = (width - line_width) / 2.0 - box[0]
        text_y = y - box[1]
        shadow = preset.get("shadow")
        if shadow:
            offset_x, offset_y = preset["shadow_offset"]
            draw.text((x + offset_x, text_y + offset_y), line, font=font, fill=shadow)
        draw.text(
            (x, text_y),
            line,
            font=font,
            fill=preset["fill"],
            stroke_width=text_stroke,
            stroke_fill=preset["stroke_fill"],
        )
        y += line_height + gap
    canvas.save(path)


def audio_duration(path: Path, ffprobe: str) -> float:
    try:
        with wave.open(str(path), "rb") as source:
            return source.getnframes() / float(source.getframerate())
    except (wave.Error, EOFError):
        result = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            errors="replace",
        )
        return float(result.stdout.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Force-align a Chinese ground-truth transcript, create SRT, and burn styled captions into MP4."
    )
    background = parser.add_mutually_exclusive_group(required=True)
    background.add_argument("--image", help="Still image used as the video picture")
    background.add_argument("--background-video", help="Loop/crop a moving background video to fit the dialogue")
    parser.add_argument("--audio", required=True, help="Audio file to mux into the video")
    parser.add_argument("--transcript-file", required=True, help="UTF-8 source transcript; wording is preserved in captions")
    parser.add_argument("--pinyin-map", help="Optional JSON pronunciation overrides, e.g. {\"行\": \"xing\"}")
    parser.add_argument("--output", required=True, help="Output MP4 file")
    parser.add_argument("--srt", help="Optional SRT path; defaults to the output filename with .srt")
    parser.add_argument("--alignment-json", help="Optional per-character timing and confidence report")
    parser.add_argument("--aligner-host", default="a100-whn", help="SSH host with the CPU alignment environment (default: a100-whn)")
    parser.add_argument("--remote-python", default="/data3/whn/Project-voca/.venv-qwen/bin/python", help="Python containing torch, torchaudio, and pypinyin on the remote host")
    parser.add_argument("--remote-cache", default="/data3/whn/Project-voca/.cache/mms-fa", help="Persistent remote cache for the MMS_FA model weights")
    parser.add_argument("--threads", type=int, default=8, help="CPU threads used by MMS_FA on the remote host")
    parser.add_argument("--ffmpeg", help="Path to ffmpeg; auto-detected if omitted")
    parser.add_argument("--font", help="Font file with Chinese glyphs; macOS system font is auto-detected")
    parser.add_argument("--long-edge", type=int, default=2048, help="Output long edge in pixels (default: 2048)")
    parser.add_argument("--sharpen", type=float, default=0.25, help="Mild luma sharpening after upscaling (0 disables it)")
    parser.add_argument("--max-chars", type=int, default=18, help="Maximum spoken characters per caption card")
    parser.add_argument("--chars-per-line", type=int, default=9, help="Prefer line breaks between phrases after this many spoken characters")
    parser.add_argument("--max-cue-duration", type=float, default=4.5, help="Do not combine phrases into a subtitle card longer than this")
    parser.add_argument("--pad-before", type=float, default=0.04)
    parser.add_argument("--pad-after", type=float, default=0.15)
    parser.add_argument("--font-size", type=int, default=192)
    parser.add_argument("--stroke-width", type=int, default=13)
    parser.add_argument("--margin-bottom", type=int, default=120)
    parser.add_argument("--subtitle-style", choices=SUBTITLE_STYLES, default="outline",
                        help="Caption preset: outline, dark-rectangle, rounded-box, yellow-highlight, or shadow")
    parser.add_argument("--box-opacity", type=int, default=84,
                        help="Background opacity percentage for the two box styles (0-100)")
    parser.add_argument("--normalize-audio", action="store_true", help="Normalize dialogue to -16 LUFS; off by default")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing video and sidecar outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.long_edge < 128:
        raise ValueError("--long-edge must be at least 128 pixels.")
    if args.max_chars < 1 or args.chars_per_line < 1 or args.max_cue_duration <= 0:
        raise ValueError("Subtitle character limits and maximum duration must be positive.")
    if not 0 <= args.sharpen <= 1.5:
        raise ValueError("--sharpen must be between 0 and 1.5.")
    if not 0 <= args.box_opacity <= 100:
        raise ValueError("--box-opacity must be between 0 and 100.")
    if args.font_size < 1 or args.stroke_width < 0 or args.margin_bottom < 0:
        raise ValueError("Font size must be positive; stroke width and bottom margin cannot be negative.")
    image_path = Path(args.image).expanduser().resolve() if args.image else None
    background_video_path = Path(args.background_video).expanduser().resolve() if args.background_video else None
    audio_path = Path(args.audio).expanduser().resolve()
    transcript_path = Path(args.transcript_file).expanduser().resolve()
    pinyin_map_path = Path(args.pinyin_map).expanduser().resolve() if args.pinyin_map else None
    output_path = Path(args.output).expanduser().resolve()
    srt_path = Path(args.srt).expanduser().resolve() if args.srt else output_path.with_suffix(".srt")
    alignment_json_path = (
        Path(args.alignment_json).expanduser().resolve()
        if args.alignment_json
        else output_path.with_suffix(".alignment.json")
    )
    if not args.overwrite:
        existing = [path for path in (output_path, srt_path, alignment_json_path) if path.exists()]
        if existing:
            raise FileExistsError(
                "Output already exists; pass --overwrite to replace: {}".format(", ".join(str(path) for path in existing))
            )
    for path in (image_path, background_video_path, audio_path, transcript_path):
        if path is None:
            continue
        if not path.is_file():
            raise FileNotFoundError("Input file not found: {}".format(path))
    if pinyin_map_path and not pinyin_map_path.is_file():
        raise FileNotFoundError("Pinyin map not found: {}".format(pinyin_map_path))
    if output_path.suffix.lower() != ".mp4":
        raise ValueError("Output must use the .mp4 extension.")

    ffmpeg = args.ffmpeg or find_executable(
        "ffmpeg", ["/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg", "/opt/local/bin/ffmpeg"]
    )
    ffprobe = str(Path(ffmpeg).with_name("ffprobe"))
    if not Path(ffprobe).is_file():
        ffprobe = find_executable("ffprobe", ["/opt/homebrew/bin/ffprobe", "/usr/local/bin/ffprobe"])
    font_path = find_font(args.font)

    transcript = transcript_path.read_text(encoding="utf-8-sig").strip()
    if not transcript:
        raise ValueError("Transcript file is empty.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    srt_path.parent.mkdir(parents=True, exist_ok=True)
    alignment_json_path.parent.mkdir(parents=True, exist_ok=True)

    if image_path:
        with Image.open(image_path) as still:
            source_width, source_height = still.size
    else:
        probe_result = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "json", str(background_video_path)],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8", errors="replace",
        )
        probe_streams = json.loads(probe_result.stdout).get("streams", [])
        if not probe_streams:
            raise ValueError("Background video has no video stream.")
        source_width, source_height = int(probe_streams[0]["width"]), int(probe_streams[0]["height"])
    scale = args.long_edge / float(max(source_width, source_height))
    width = int(round(source_width * scale))
    height = int(round(source_height * scale))
    width += width % 2
    height += height % 2
    font = ImageFont.truetype(font_path, args.font_size, index=0)
    duration = audio_duration(audio_path, ffprobe)

    helper_path = Path(__file__).with_name("force_align_chinese.py")
    aligner_result = run_forced_alignment(
        audio_path,
        transcript_path,
        pinyin_map_path,
        helper_path,
        args.aligner_host,
        args.remote_python,
        args.remote_cache,
        args.threads,
    )
    alignment_json_path.write_text(
        json.dumps(aligner_result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    aligned: List[Optional[TimedChar]] = [None] * sum(1 for char in transcript if speech_char(char))
    for item in aligner_result["units"]:
        aligned[item["index"]] = TimedChar(
            item["char"], item["start"], item["end"], item["confidence"]
        )
    if any(item is None for item in aligned):
        raise RuntimeError("Forced aligner did not return a time for every spoken transcript character.")

    with tempfile.TemporaryDirectory(prefix="chinese-subtitle-align-") as temp_dir:
        chunks = split_chunks(
            transcript,
            aligned,
            args.max_chars,
            args.chars_per_line,
            0.5,
            args.max_cue_duration,
        )
        cues = make_cues(chunks, aligned, duration, args.pad_before, args.pad_after)
        write_srt(cues, srt_path)

        overlays: List[Path] = []
        for index, cue in enumerate(cues, 1):
            overlay_path = Path(temp_dir) / "caption-{:03}.png".format(index)
            render_caption(
                cue.text, overlay_path, width, height, font,
                args.stroke_width, args.margin_bottom, args.subtitle_style, args.box_opacity,
            )
            overlays.append(overlay_path)

        ffmpeg_command = [ffmpeg, "-hide_banner", "-y", "-filter_complex_threads", "1"]
        if image_path:
            ffmpeg_command += ["-loop", "1", "-framerate", "30", "-i", str(image_path)]
        else:
            ffmpeg_command += ["-stream_loop", "-1", "-i", str(background_video_path)]
        ffmpeg_command += ["-i", str(audio_path)]
        for overlay_path in overlays:
            ffmpeg_command += ["-loop", "1", "-framerate", "30", "-i", str(overlay_path)]

        base_filter = "scale={}:{}:flags=lanczos".format(width, height)
        if args.sharpen:
            base_filter += ",unsharp=5:5:{:.2f}:5:5:0.0".format(args.sharpen)
        base_filter += ",setsar=1"
        graph = ["[0:v]{}[base]".format(base_filter)]
        previous = "[base]"
        for index, cue in enumerate(cues, 1):
            next_label = "[v{}]".format(index)
            input_index = index + 1
            graph.append(
                "{}[{}:v]overlay=0:0:format=auto:enable='between(t,{:.3f},{:.3f})'{}".format(
                    previous, input_index, cue.start, cue.end, next_label
                )
            )
            previous = next_label

        ffmpeg_command += [
            "-filter_complex", ";".join(graph),
            "-map", previous,
            "-map", "1:a:0",
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-r", "30",
            "-c:a", "aac",
            "-profile:a", "aac_low",
            "-b:a", "192k",
            "-ar", "48000",
            "-ac", "2",
            "-disposition:a:0", "default",
            "-t", "{:.6f}".format(duration),
            "-shortest",
            "-movflags", "+faststart",
        ]
        if image_path:
            ffmpeg_command[ffmpeg_command.index("-crf"):ffmpeg_command.index("-crf")] = ["-tune", "stillimage"]
        if args.normalize_audio:
            ffmpeg_command += ["-af", "loudnorm=I=-16:TP=-1.5:LRA=11"]
        ffmpeg_command.append(str(output_path))

        ffmpeg_result = subprocess.run(
            ffmpeg_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            errors="replace",
        )
        if ffmpeg_result.returncode:
            raise RuntimeError("ffmpeg failed:\n{}".format(ffmpeg_result.stderr[-5000:]))

    print(
        "MMS_FA forced-alignment mean confidence: {:.1%}".format(
            aligner_result["mean_confidence"]
        )
    )
    low_confidence = [item for item in aligner_result["units"] if item["confidence"] < 0.15]
    if low_confidence:
        details = " ".join(
            "{}({}):{:.0%}".format(
                item["char"], item["pronunciation"], item["confidence"]
            )
            for item in low_confidence
        )
        print("Review low-confidence character alignments (<15%): {}".format(details))
    trimmed = [item for item in aligner_result["units"] if item.get("trimmed_silence")]
    if trimmed:
        print("Trimmed long silent gaps from {} forced-alignment span(s).".format(len(trimmed)))
    print("Embedded {} transcript-grounded subtitle cues.".format(len(cues)))
    print("Subtitle style: {} (box opacity {}%).".format(args.subtitle_style, args.box_opacity))
    print("Video resolution: {}x{} (long edge {} pixels).".format(width, height, args.long_edge))
    print("Video: {}".format(output_path))
    print("SRT:   {}".format(srt_path))
    print("Timing report: {}".format(alignment_json_path))


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print("error: {}".format(error), file=sys.stderr)
        sys.exit(2)
