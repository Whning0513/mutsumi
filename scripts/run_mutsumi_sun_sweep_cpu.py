#!/usr/bin/env python3
"""Generate the Qwen sun speech source and a selectable set of RVC variants."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


VARIANTS = {
    "i000_p-1": (0.00, -1),
    "i015_p0": (0.15, 0),
    "i015_p-1": (0.15, -1),
    "i015_p-2": (0.15, -2),
    "i030_p-1": (0.30, -1),
    "i045_p-1": (0.45, -1),
}


def main() -> None:
    default_root = Path(os.environ.get("VOCA_ROOT", Path(__file__).resolve().parent.parent))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=default_root)
    parser.add_argument("--variants", help="Comma-separated variant labels; default runs all six")
    parser.add_argument("--source-only", action="store_true", help="Generate only the Qwen source WAV")
    parser.add_argument("--skip-source", action="store_true", help="Reuse an existing Qwen source WAV")
    parser.add_argument("--dry-run", action="store_true", help="Print work without loading models or writing files")
    parser.add_argument("--overwrite", action="store_true", help="Replace outputs with matching names")
    parser.add_argument("--text-file", type=Path)
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    framework = root / "frameworks/RVC"
    qwen_python = root / ".venv-qwen/bin/python"
    rvc_python = root / ".venv-rvc-cpu/bin/python"
    model = root / "models/archive/qwen3-tts-bang-dream"
    rvc_model = root / "models/rvc/mutsumi-rvc/mutsumi.pth"
    rvc_index = root / "models/rvc/mutsumi-rvc/added_IVF345_Flat_nprobe_1_mutsumi_v2.index"
    text_path = (args.text_file or root / "scripts/若叶睦_太阳宣言_测试文本.txt").expanduser().resolve()
    output_dir = root / "outputs/试听试验"
    source_path = output_dir / "若叶睦_Qwen_太阳宣言_原声.wav"
    selected = list(VARIANTS) if not args.variants else [item.strip() for item in args.variants.split(",") if item.strip()]
    unknown = sorted(set(selected) - set(VARIANTS))
    if unknown:
        parser.error("unknown variant(s): " + ", ".join(unknown))
    if len(selected) != len(set(selected)):
        parser.error("variant labels must not repeat")

    source_command = [str(qwen_python), str(root / "scripts/generate_qwen_voice_cpu.py"),
                      "--root", str(root), "--model", str(model), "--speaker", "mujica_mutsumi",
                      "--text-file", str(text_path), "--output", str(source_path)]
    print("Source command:", " ".join(source_command), flush=True)
    if not args.skip_source and not args.dry_run:
        if source_path.exists() and not args.overwrite:
            raise FileExistsError(f"Source already exists; pass --overwrite: {source_path}")
        env = os.environ.copy()
        env.update({"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "32", "MKL_NUM_THREADS": "32", "OPENBLAS_NUM_THREADS": "1"})
        subprocess.run(source_command, cwd=root, env=env, check=True)
    elif args.skip_source and not source_path.is_file() and not args.dry_run:
        raise FileNotFoundError(f"--skip-source requires an existing source: {source_path}")
    if args.source_only:
        if args.dry_run:
            print("Dry run; source only, no files written.")
        return
    if args.dry_run:
        print("Dry run; variant commands will be listed but not executed.")

    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "32", "MKL_NUM_THREADS": "32",
        "OPENBLAS_NUM_THREADS": "1", "RVC_AUDIO_FORCE_CPU": "1", "rmvpe_root": str(root / "assets/rvc"),
        "PYTHONPATH": ":".join([str(framework), str(root / ".venv-qwen/lib/python3.10/site-packages"),
                                  str(root / ".venv-rvc-cpu/lib/python3.10/site-packages")]),
    })
    for label in selected:
        index_rate, pitch = VARIANTS[label]
        output_path = output_dir / f"若叶睦_RVC_太阳宣言_protect050_index{int(index_rate * 100):03d}_pitch{pitch}.wav"
        if output_path.exists() and not args.overwrite and not args.dry_run:
            raise FileExistsError(f"Variant already exists; pass --overwrite: {output_path}")
        command = [str(rvc_python), str(framework / "infer/cli.py"), "--model", str(rvc_model),
                   "--input", str(source_path), "--output", str(output_path), "--index", str(rvc_index),
                   "--index-rate", str(index_rate), "--rms-mix-rate", "0.45", "--protect", "0.50",
                   "--pitch", str(pitch), "--f0-method", "rmvpe"]
        if args.overwrite:
            command.append("--overwrite")
        print(f"\nRVC variant {label}: {' '.join(command)}", flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=framework, env=env, check=True)
    if args.dry_run:
        print("No files written.")


if __name__ == "__main__":
    main()
