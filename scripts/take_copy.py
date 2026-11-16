#!/usr/bin/env python3
"""Select one explicitly marked copy block, format it, then mark it as used."""

import argparse
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POOL = ROOT / "inputs" / "原始文本" / "文案池.txt"
HEADER = re.compile(r"^@@@ id=([A-Za-z0-9][A-Za-z0-9_-]*) status=(待取|已取) @@@$")
END_MARKER = "@@@ end @@@"


@dataclass
class CopySegment:
    segment_id: str
    status: str
    text: str
    header_index: int


def parse_pool(pool_path: Path):
    if not pool_path.is_file():
        raise FileNotFoundError(
            "Copy pool not found: {}. Create it and add explicitly marked blocks.".format(pool_path)
        )
    lines = pool_path.read_bytes().decode("utf-8").splitlines(keepends=True)
    segments: List[CopySegment] = []
    ids = set()
    current = None
    content: List[str] = []

    for line_index, line in enumerate(lines):
        value = line.rstrip("\r\n")
        if line_index == 0 and value.startswith("\ufeff"):
            value = value[1:]
        if current is None:
            if not value.strip():
                continue
            match = HEADER.fullmatch(value)
            if not match:
                raise ValueError(
                    "Unexpected text outside a marked block at line {}. "
                    "Start each item with an @@@ id=... status=待取 @@@ header.".format(line_index + 1)
                )
            segment_id, status = match.groups()
            if segment_id in ids:
                raise ValueError("Duplicate copy ID: {}".format(segment_id))
            ids.add(segment_id)
            current = (segment_id, status, line_index)
            content = []
            continue

        if value == END_MARKER:
            segment_id, status, header_index = current
            text = "".join(content).strip()
            if not text:
                raise ValueError("Copy block {} is empty.".format(segment_id))
            segments.append(CopySegment(segment_id, status, text, header_index))
            current = None
            content = []
            continue
        if HEADER.fullmatch(value):
            raise ValueError(
                "Copy block {} has no {} marker before the next header.".format(
                    current[0], END_MARKER
                )
            )
        content.append(line)

    if current is not None:
        raise ValueError("Copy block {} has no {} marker.".format(current[0], END_MARKER))
    return lines, segments


def command_list(args: argparse.Namespace) -> None:
    _, segments = parse_pool(Path(args.pool).expanduser().resolve())
    if not segments:
        print("No copy blocks yet. Add marked blocks to {}.".format(args.pool))
        return
    print("ID\t状态\t字符数")
    for segment in segments:
        print("{}\t{}\t{}".format(segment.segment_id, segment.status, len(segment.text)))


def mark_as_used(pool_path: Path, lines: List[str], segment: CopySegment) -> None:
    original = lines[segment.header_index]
    newline = "\r\n" if original.endswith("\r\n") else "\n" if original.endswith("\n") else ""
    bom = "\ufeff" if segment.header_index == 0 and original.startswith("\ufeff") else ""
    lines[segment.header_index] = "{}@@@ id={} status=已取 @@@{}".format(bom, segment.segment_id, newline)
    temporary = pool_path.with_name(pool_path.name + ".tmp")
    temporary.write_bytes("".join(lines).encode("utf-8"))
    temporary.replace(pool_path)


def command_take(args: argparse.Namespace) -> None:
    pool_path = Path(args.pool).expanduser().resolve()
    _, segments = parse_pool(pool_path)
    matches = [segment for segment in segments if segment.segment_id == args.segment_id]
    if not matches:
        raise ValueError("No copy block with ID {}.".format(args.segment_id))
    segment = matches[0]
    if segment.status != "待取":
        raise ValueError("Copy block {} is already marked 已取.".format(segment.segment_id))

    output = Path(args.output).expanduser() if args.output else ROOT / "inputs" / "台本" / (segment.segment_id + ".txt")
    output = output.resolve()
    chunks_dir = Path(args.chunks_dir).expanduser() if args.chunks_dir else ROOT / "outputs" / "文案切块" / segment.segment_id
    chunks_dir = chunks_dir.resolve()
    manifest_path = output.with_suffix(".chunks.json")
    if output.exists():
        raise FileExistsError("Output already exists: {}".format(output))
    if manifest_path.exists():
        raise FileExistsError("Chunk manifest already exists: {}".format(manifest_path))
    if chunks_dir.exists():
        raise FileExistsError("Chunk directory already exists: {}".format(chunks_dir))

    command = [
        sys.executable,
        str(Path(__file__).with_name("prepare_tts_text.py")),
        "-",
        "--mode", args.mode,
        "--split-mode", "fixed",
        "--max-chars", str(args.max_chars),
        "--output", str(output),
        "--chunks-dir", str(chunks_dir),
    ]
    if args.mode == "rewrite":
        command += ["--style", args.style]
    try:
        subprocess.run(command, input=segment.text, encoding="utf-8", check=True)
    except subprocess.CalledProcessError:
        for path in (output, output.with_name(output.name + ".tmp"), manifest_path,
                     manifest_path.with_name(manifest_path.name + ".tmp")):
            path.unlink(missing_ok=True)
        shutil.rmtree(chunks_dir, ignore_errors=True)
        raise

    # Re-read after formatting so edits or newly appended blocks are preserved.
    latest_lines, latest_segments = parse_pool(pool_path)
    latest_matches = [item for item in latest_segments if item.segment_id == segment.segment_id]
    if not latest_matches or latest_matches[0].status != "待取":
        raise ValueError("Copy pool changed while formatting; output exists but the block was not marked.")
    if latest_matches[0].text != segment.text:
        raise ValueError("Selected copy changed while formatting; output exists but the block was not marked.")
    mark_as_used(pool_path, latest_lines, latest_matches[0])
    print("Marked {} as 已取.".format(segment.segment_id))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="List or take explicitly segmented copy from the local copy pool."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List IDs, status, and character counts")
    list_parser.add_argument("--pool", default=str(DEFAULT_POOL))
    list_parser.set_defaults(handler=command_list)

    take_parser = subparsers.add_parser("take", help="Format exactly one available block and mark it used")
    take_parser.add_argument("segment_id", help="Exact ID from the copy pool; no sentence-based selection")
    take_parser.add_argument("--pool", default=str(DEFAULT_POOL))
    take_parser.add_argument("--output", help="TTS-ready 台本 path (default: inputs/台本/<ID>.txt)")
    take_parser.add_argument("--chunks-dir", help="Fixed-length chunk directory (default: outputs/文案切块/<ID>)")
    take_parser.add_argument("--max-chars", type=int, default=220)
    take_parser.add_argument("--mode", choices=("clean", "rewrite"), default="clean")
    take_parser.add_argument(
        "--style",
        default="中文搞笑口播；保留角色口吻、夸张感和原有笑点，读起来顺口自然",
        help="Style guidance when --mode rewrite is selected",
    )
    take_parser.set_defaults(handler=command_take)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        if args.command == "take" and args.max_chars < 10:
            raise ValueError("--max-chars must be at least 10.")
        args.handler(args)
    except (FileNotFoundError, FileExistsError, OSError, ValueError, subprocess.CalledProcessError) as error:
        print("error: {}".format(error), file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
