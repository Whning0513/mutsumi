from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import quote

from .config import ProjectConfig, load_config, load_profiles
from .audio import DEFAULT_OUTPUT_GAIN, DEFAULT_PEAK_LIMIT_DBFS, apply_gain_to_file, gain_db


MEDIA_SUFFIXES = {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".mp4", ".mov", ".mkv", ".webm"}


def _which(name: str) -> str | None:
    return shutil.which(name)


def _run(command: Sequence[str], *, cwd: Path | None = None) -> None:
    print("$ " + shlex.join(str(item) for item in command), flush=True)
    subprocess.run(list(command), cwd=cwd, check=True)


def _remote_run(config: ProjectConfig, command: str) -> None:
    host = str(config.remote.get("host", ""))
    if not host:
        raise ValueError("Set [remote].host in the project config first")
    _run(["ssh", "-T", host, command])


def _remote_shell(config: ProjectConfig, body: str) -> str:
    root = str(config.remote["root"])
    return f"cd {shlex.quote(root)} && {body}"


def _env_command(assignments: dict[str, str], argv: Sequence[str]) -> str:
    env = " ".join(f"{key}={shlex.quote(value)}" for key, value in assignments.items())
    return f"env {env} " + shlex.join(str(item) for item in argv)


def _remote_python(config: ProjectConfig, key: str, default: str) -> str:
    rel = str(config.remote.get(key, default))
    return config.remote_path(rel)


def _profile(config: ProjectConfig, name: str) -> dict[str, Any]:
    profiles = load_profiles(config)
    if name not in profiles:
        raise ValueError(f"Unknown voice profile {name!r}; choose from: {', '.join(profiles)}")
    return profiles[name]


def _asset_path(config: ProjectConfig, value: str) -> str:
    return config.remote_path(value)


def synthesize(
    config: ProjectConfig,
    profile_name: str,
    text: str,
    output: Path,
    overrides: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> None:
    profile = dict(_profile(config, profile_name))
    profile.update({key: value for key, value in (overrides or {}).items() if value is not None})
    route = profile.get("route")
    if route not in {"gsv", "qwen_rvc"}:
        raise ValueError(f"Unsupported profile route: {route!r}")
    if not text.strip():
        raise ValueError("Synthesis text is empty")
    audio_config = config.values.get("audio", {})
    output_gain = float(profile.get("output_gain", audio_config.get("output_gain", DEFAULT_OUTPUT_GAIN)))
    peak_limit_dbfs = float(audio_config.get("peak_limit_dbfs", DEFAULT_PEAK_LIMIT_DBFS))
    gain_db(output_gain)
    output = output.expanduser().resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output already exists (pass --overwrite to replace it): {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    host = str(config.remote.get("host", ""))
    if not host:
        raise ValueError("Set [remote].host in the project config first")
    stage = f"/tmp/voca-{uuid.uuid4().hex[:12]}"
    remote_text = stage + "/text.txt"
    remote_output = stage + "/result.wav"
    with tempfile.TemporaryDirectory(prefix="voca-") as local_temp:
        text_path = Path(local_temp) / "text.txt"
        text_path.write_text(text.strip() + "\n", encoding="utf-8")
        _run(["ssh", "-T", host, f"mkdir -p {shlex.quote(stage)}"])
        try:
            _run(["scp", str(text_path), f"{host}:{remote_text}"])
            if route == "gsv":
                script = config.remote_path("scripts/generate_gsv_zeroshot_cpu.py")
                python = _remote_python(config, "gsv_python", ".venv-gpt-sovits-cpu/bin/python")
                root = str(config.remote["root"]).rstrip("/")
                gsv_framework = _asset_path(config, config.remote["models"]["gsv_root"])
                command = [
                    python, script, "--root", root, "--pretrained-root", _asset_path(config, config.remote["models"].get("gsv_pretrained", "models/gpt-sovits/pretrained_models")), "--text-file", remote_text,
                    "--reference", _asset_path(config, profile["reference_audio"]),
                    "--prompt-text-file", _asset_path(config, profile["reference_prompt_file"]), "--output", remote_output,
                    "--sovits-weight", _asset_path(config, profile["sovits_weight"]),
                    "--gpt-weight", _asset_path(config, profile["gpt_weight"]),
                    "--top-k", str(profile.get("top_k", 15)), "--top-p", str(profile.get("top_p", 0.6)),
                    "--temperature", str(profile.get("temperature", 0.6)), "--speed", str(profile.get("speed", 1.0)),
                    "--target-rms-dbfs", str(profile.get("target_rms_dbfs", -27.0)),
                    "--output-gain", "1.0",
                ]
                assignments = {
                    "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": str(config.runtime.get("threads", 32)),
                    "MKL_NUM_THREADS": str(config.runtime.get("threads", 32)), "OPENBLAS_NUM_THREADS": "1",
                    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                    "PYTHONPATH": f"{gsv_framework}:{gsv_framework}/GPT_SoVITS:{_asset_path(config, '.venv-qwen/lib/python' + str(config.remote.get('python_site_version', '3.10')) + '/site-packages')}",
                }
            else:
                qwen_python = _remote_python(config, "qwen_python", ".venv-qwen/bin/python")
                rvc_python = _remote_python(config, "rvc_python", ".venv-rvc-cpu/bin/python")
                root = str(config.remote["root"]).rstrip("/")
                source = stage + "/source.wav"
                qwen_script = config.remote_path("scripts/generate_qwen_voice_cpu.py")
                qwen_model = _asset_path(config, config.remote["models"].get("qwen", "models/archive/qwen3-tts-bang-dream"))
                rvc_framework = _asset_path(config, "frameworks/RVC")
                rvc_script = rvc_framework + "/infer/cli.py"
                rvc_model = _asset_path(config, config.remote["models"]["rvc_model"])
                rvc_index = _asset_path(config, config.remote["models"]["rvc_index"])
                qwen_cmd = [qwen_python, qwen_script, "--root", root, "--model", qwen_model,
                            "--speaker", str(profile.get("qwen_speaker", "mujica_mutsumi")),
                            "--text-file", remote_text, "--output", source, "--output-gain", "1.0"]
                _remote_run(config, _remote_shell(config, _env_command({
                    "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": str(config.runtime.get("threads", 32)),
                    "MKL_NUM_THREADS": str(config.runtime.get("threads", 32)), "OPENBLAS_NUM_THREADS": "1",
                }, qwen_cmd)))
                rvc_cmd = [rvc_python, rvc_script, "--model", rvc_model, "--input", source,
                           "--output", remote_output, "--index", rvc_index,
                           "--index-rate", str(profile.get("index_rate", 0.15)),
                           "--rms-mix-rate", str(profile.get("rms_mix_rate", 0.45)),
                           "--protect", str(profile.get("protect", 0.45)),
                           "--pitch", str(profile.get("pitch", 0)),
                           "--f0-method", str(profile.get("f0_method", "rmvpe"))]
                assignments = {
                    "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": str(config.runtime.get("threads", 32)),
                    "MKL_NUM_THREADS": str(config.runtime.get("threads", 32)), "OPENBLAS_NUM_THREADS": "1",
                    "RVC_AUDIO_FORCE_CPU": "1", "rmvpe_root": _asset_path(config, "assets/rvc"),
                    "PYTHONPATH": f"{rvc_framework}:{root}/.venv-qwen/lib/python{config.remote.get('python_site_version', '3.10')}/site-packages:{root}/.venv-rvc-cpu/lib/python{config.remote.get('python_site_version', '3.10')}/site-packages",
                }
                _remote_run(config, _remote_shell(config, _env_command(assignments, rvc_cmd)))
            if route == "gsv":
                _remote_run(config, _remote_shell(config, _env_command(assignments, command)))
            _run(["scp", f"{host}:{remote_output}", str(output)])
            apply_gain_to_file(output, output_gain, peak_limit_dbfs)
            print(f"Applied output gain ×{output_gain:g} ({gain_db(output_gain):+.2f} dB), peak limit {peak_limit_dbfs:g} dBFS")
        finally:
            subprocess.run(["ssh", "-T", host, f"rm -rf {shlex.quote(stage)}"], check=False)
    print(f"Saved {output}")


def find_ffprobe() -> str:
    executable = _which("ffprobe")
    if not executable:
        raise FileNotFoundError("ffprobe is required; install FFmpeg and make ffprobe available on PATH")
    return executable


def probe_media(path: Path) -> dict[str, Any]:
    command = [find_ffprobe(), "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command_init_config(config: ProjectConfig) -> None:
    destination = config.root / "config/local.toml"
    if destination.exists():
        raise FileExistsError(f"Local config already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config.root / "configs/project.example.toml", destination)
    print(f"Created {destination}; edit [remote].host/root for your machine")


def command_doctor(config: ProjectConfig, remote: bool) -> None:
    print(f"Project root: {config.root}")
    print(f"Config:       {config.file}")
    print(f"Remote:       {config.remote.get('host', '(unset)')}:{config.remote.get('root', '(unset)')}")
    profiles = load_profiles(config)
    print(f"Profiles:     {len(profiles)} ({', '.join(profiles)})")
    for tool in ("git", "ssh", "scp", "ffmpeg", "ffprobe"):
        print(f"{tool:12} {_which(tool) or 'not found'}")
    missing = [item for item in ("scripts/generate_gsv_zeroshot_cpu.py", "scripts/embed_chinese_subtitles.py", "configs/voice_profiles.toml") if not (config.root / item).is_file()]
    if missing:
        print("Missing project files: " + ", ".join(missing))
    if remote:
        root = str(config.remote["root"])
        checks = [
            f"test -d {shlex.quote(root)}",
            f"test -x {shlex.quote(root + '/.venv-qwen/bin/python')}",
            f"test -x {shlex.quote(root + '/.venv-rvc-cpu/bin/python')}",
            f"test -x {shlex.quote(root + '/.venv-gpt-sovits-cpu/bin/python')}",
        ]
        command = " && ".join(checks) + " && echo 'remote project root and CPU environments: OK'"
        _remote_run(config, command)


def command_env_snapshot(args: argparse.Namespace, config: ProjectConfig) -> None:
    host = str(config.remote.get("host", ""))
    if not host:
        raise ValueError("Set [remote].host in the project config first")
    root = str(config.remote["root"]).rstrip("/")
    environments = {
        "qwen": str(config.remote.get("qwen_python", ".venv-qwen/bin/python")),
        "rvc_cpu": str(config.remote.get("rvc_python", ".venv-rvc-cpu/bin/python")),
        "gsv_cpu": str(config.remote.get("gsv_python", ".venv-gpt-sovits-cpu/bin/python")),
    }
    snapshot: dict[str, Any] = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "host": host,
        "remote_root": root,
        "local": {"python": sys.version, "platform": sys.platform},
        "remote_environments": {},
    }
    local_freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
    if local_freeze.returncode == 0:
        snapshot["local"]["pip_freeze"] = local_freeze.stdout.splitlines()
    for name, relative_python in environments.items():
        python = config.remote_path(relative_python)
        prefix = f"cd {shlex.quote(root)} && "
        version_result = subprocess.run(["ssh", "-T", host, prefix + shlex.join([python, "--version"])],
                                        capture_output=True, text=True)
        freeze_result = subprocess.run(["ssh", "-T", host, prefix + shlex.join([python, "-m", "pip", "freeze"])],
                                       capture_output=True, text=True)
        if version_result.returncode or freeze_result.returncode:
            detail = (version_result.stderr + "\n" + freeze_result.stderr).strip()
            raise RuntimeError(f"Could not snapshot remote {name} environment: {detail}")
        snapshot["remote_environments"][name] = {
            "python": version_result.stdout.strip(),
            "pip_freeze": freeze_result.stdout.splitlines(),
        }
    destination = Path(args.output).expanduser().resolve() if args.output else config.root / "outputs/manifests" / f"cpu-env-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Environment snapshot: {destination}")


def command_profiles(config: ProjectConfig, name: str | None) -> None:
    profiles = load_profiles(config)
    selected = [name] if name else list(profiles)
    for key in selected:
        if key not in profiles:
            raise ValueError(f"Unknown profile {key!r}")
        item = profiles[key]
        print(f"{key}\n  {item.get('name', '')} — {item.get('description', '')}\n  route={item.get('route')}")
        for field in ("index_rate", "pitch", "protect", "rms_mix_rate", "top_k", "top_p", "temperature", "speed", "target_rms_dbfs"):
            if field in item:
                print(f"  {field}={item[field]}")


def command_manifest(args: argparse.Namespace, config: ProjectConfig) -> None:
    path = Path(args.media).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest = {
        "schema_version": 1,
        "file": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _hash_file(path),
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "profile": args.profile,
        "parameters": json.loads(args.parameters) if args.parameters else {},
    }
    try:
        manifest["media"] = probe_media(path)
    except FileNotFoundError as error:
        manifest["media_probe_unavailable"] = str(error)
    destination = Path(args.output).expanduser().resolve() if args.output else config.root / "outputs/manifests" / (path.name + ".json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Manifest: {destination}")


def command_index(args: argparse.Namespace, config: ProjectConfig) -> None:
    root = Path(args.root).expanduser().resolve() if args.root else config.root
    if not root.is_dir():
        raise NotADirectoryError(root)
    output = Path(args.output).expanduser().resolve() if args.output else config.root / "outputs/asset-index.csv"
    rows: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item.suffix.lower() in MEDIA_SUFFIXES):
        try:
            data = probe_media(path)
            rows.append({"path": str(path.relative_to(root)), "size_bytes": path.stat().st_size,
                         "duration": data.get("format", {}).get("duration", ""),
                         "streams": ",".join(stream.get("codec_type", "") for stream in data.get("streams", []))})
        except (subprocess.CalledProcessError, json.JSONDecodeError) as error:
            rows.append({"path": str(path.relative_to(root)), "size_bytes": path.stat().st_size, "duration": "", "streams": f"probe_error:{error}"})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "size_bytes", "duration", "streams"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Indexed {len(rows)} media files: {output}")


def command_qc(args: argparse.Namespace) -> None:
    path = Path(args.media).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    data = probe_media(path)
    format_info = data.get("format", {})
    print(f"File: {path}\nSize: {path.stat().st_size:,} bytes")
    if "duration" in format_info:
        print(f"Duration: {float(format_info['duration']):.3f} s")
    for number, stream in enumerate(data.get("streams", []), 1):
        common = [f"type={stream.get('codec_type')}", f"codec={stream.get('codec_name')}"]
        if stream.get("codec_type") == "audio":
            common += [f"rate={stream.get('sample_rate')} Hz", f"channels={stream.get('channels')}"]
        if stream.get("codec_type") == "video":
            common += [f"size={stream.get('width')}x{stream.get('height')}", f"fps={stream.get('r_frame_rate')}"]
        print(f"Stream {number}: " + "; ".join(common))
    if not data.get("streams"):
        raise ValueError("No audio or video streams detected")
    audio = [item for item in data["streams"] if item.get("codec_type") == "audio"]
    if path.suffix.lower() in {".wav", ".flac"} and not audio:
        raise ValueError("Audio file has no audio stream")
    if audio:
        ffmpeg = _which("ffmpeg")
        if not ffmpeg:
            print("Loudness: ffmpeg not found; stream metadata only")
        else:
            result = subprocess.run(
                [ffmpeg, "-hide_banner", "-i", str(path), "-vn", "-af", "volumedetect", "-f", "null", "-"],
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
            )
            if result.returncode:
                raise RuntimeError(f"ffmpeg could not decode {path}: {(result.stderr or '')[-1500:]}")
            stderr = result.stderr or ""
            for label in ("mean_volume", "max_volume"):
                value = next((line.split(":", 1)[1].strip() for line in stderr.splitlines() if label in line), None)
                if value:
                    print(f"{label.replace('_', ' ').title()}: {value}")


def command_compare(args: argparse.Namespace, config: ProjectConfig) -> None:
    paths = [Path(item).expanduser().resolve() for item in args.media]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    destination = Path(args.output).expanduser().resolve() if args.output else config.root / "outputs/compare/listen.html"
    destination.parent.mkdir(parents=True, exist_ok=True)
    cards = []
    for path in paths:
        rel = quote(os.path.relpath(path, destination.parent).replace(os.sep, "/"), safe="/:")
        cards.append(f"<section><h2>{html.escape(path.name)}</h2><audio controls preload='metadata' src='{html.escape(rel, quote=True)}'></audio><p>{path.stat().st_size:,} bytes</p></section>")
    page = "<!doctype html><meta charset='utf-8'><title>Audio comparison</title><style>body{font:16px system-ui;max-width:850px;margin:40px auto;padding:0 16px}section{padding:16px 0;border-bottom:1px solid #ddd}audio{width:100%}</style><h1>Audio comparison</h1>" + "\n".join(cards)
    destination.write_text(page, encoding="utf-8")
    print(f"Comparison page: {destination}")


def command_archive(args: argparse.Namespace, config: ProjectConfig) -> None:
    source = Path(args.path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    allowed = [(config.root / "outputs/试听试验").resolve(), (config.root / "videos").resolve()]
    if not any(source == root or root in source.parents for root in allowed):
        raise ValueError("Safe archive accepts files only from outputs/试听试验 or videos")
    archive_root = (config.root / config.runtime.get("archive_dir", "outputs/archieve")).resolve()
    day = datetime.now().strftime("%Y%m%d_cli")
    outputs_root = (config.root / "outputs").resolve()
    if outputs_root == source or outputs_root in source.parents:
        relative = source.relative_to(outputs_root)
    else:
        relative = Path("videos") / source.relative_to((config.root / "videos").resolve())
    destination = archive_root / day / relative
    print(f"Source:      {source}\nDestination: {destination}")
    if not args.apply:
        print("Dry run only. Add --apply to move it.")
        return
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(destination))
    print("Moved; the file was preserved in the archive.")


def command_video(args: argparse.Namespace, config: ProjectConfig) -> None:
    script = config.root / "scripts/embed_chinese_subtitles.py"
    output = Path(args.output).expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists (pass --overwrite to replace it): {output}")
    host = args.aligner_host or str(config.remote.get("host", "a100-whn"))
    remote_python = _remote_python(config, "qwen_python", ".venv-qwen/bin/python")
    remote_cache = config.remote_path(".cache/mms-fa")
    command = [sys.executable, str(script), "--audio", str(Path(args.audio).expanduser().resolve()),
               "--transcript-file", str(Path(args.transcript_file).expanduser().resolve()),
               "--output", str(output), "--aligner-host", host,
               "--remote-python", remote_python, "--remote-cache", remote_cache,
               "--threads", str(args.threads), "--long-edge", str(args.long_edge)]
    if args.image:
        command += ["--image", str(Path(args.image).expanduser().resolve())]
    if args.background_video:
        command += ["--background-video", str(Path(args.background_video).expanduser().resolve())]
    if args.normalize_audio:
        command.append("--normalize-audio")
    if args.overwrite:
        command.append("--overwrite")
    _run(command, cwd=config.root)


def command_sweep(args: argparse.Namespace, config: ProjectConfig) -> None:
    host = str(config.remote["host"])
    script = config.remote_path("scripts/run_mutsumi_sun_sweep_cpu.py")
    python = _remote_python(config, "qwen_python", ".venv-qwen/bin/python")
    command = [python, script, "--root", str(config.remote["root"])]
    gain = args.output_gain
    if gain is None:
        gain = float(config.values.get("audio", {}).get("output_gain", DEFAULT_OUTPUT_GAIN))
    peak_limit_dbfs = args.peak_limit_dbfs
    if peak_limit_dbfs is None:
        peak_limit_dbfs = float(config.values.get("audio", {}).get("peak_limit_dbfs", DEFAULT_PEAK_LIMIT_DBFS))
    command += ["--output-gain", str(gain), "--peak-limit-dbfs", str(peak_limit_dbfs)]
    if args.variants:
        command += ["--variants", args.variants]
    if args.source_only:
        command.append("--source-only")
    if args.skip_source:
        command.append("--skip-source")
    if args.dry_run:
        command.append("--dry-run")
    if args.overwrite:
        command.append("--overwrite")
    _remote_run(config, shlex.join(command))


def command_scene_validate(args: argparse.Namespace, config: ProjectConfig) -> dict[str, Any]:
    scene_path = Path(args.scene).expanduser().resolve()
    if not scene_path.is_file() and not Path(args.scene).is_absolute():
        scene_path = (config.root / args.scene).resolve()
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    if not isinstance(scene.get("lines"), list) or not scene["lines"]:
        raise ValueError("Scene must contain a non-empty lines array")
    profiles = load_profiles(config)
    for index, line in enumerate(scene["lines"], 1):
        if not isinstance(line, dict):
            raise ValueError(f"line {index}: expected a JSON object")
        if line.get("profile") not in profiles:
            raise ValueError(f"line {index}: unknown profile {line.get('profile')!r}")
        if not isinstance(line.get("text"), str) or not line["text"].strip():
            raise ValueError(f"line {index}: text is required")
        if int(line.get("pause_after_ms", 0)) < 0:
            raise ValueError(f"line {index}: pause_after_ms cannot be negative")
    print(f"Scene OK: {scene.get('title', scene_path.stem)} ({len(scene['lines'])} lines)")
    return scene


def _canonicalize_audio(source: Path, target: Path) -> None:
    _run([_which("ffmpeg") or "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
          "-ac", "1", "-ar", "48000", "-c:a", "pcm_s16le", str(target)])


def command_scene_render(args: argparse.Namespace, config: ProjectConfig) -> None:
    scene = command_scene_validate(args, config)
    title = str(scene.get("title", Path(args.scene).stem))
    safe_title = "".join(char if char.isalnum() or char in "-_" else "_" for char in title).strip("_") or "scene"
    audio_output = Path(args.output).expanduser().resolve() if args.output else config.root / "outputs/试听试验" / f"{safe_title}.wav"
    text_output = audio_output.with_suffix(".txt")
    video_output = Path(args.video_output).expanduser().resolve() if args.video_output else config.root / "videos" / f"{safe_title}.mp4"
    if audio_output.exists() and not args.overwrite:
        raise FileExistsError(f"Audio already exists (pass --overwrite to replace it): {audio_output}")
    if text_output.exists() and not args.overwrite:
        raise FileExistsError(f"Transcript already exists (pass --overwrite to replace it): {text_output}")
    if (args.image or args.background_video) and video_output.exists() and not args.overwrite:
        raise FileExistsError(f"Video already exists (pass --overwrite to replace it): {video_output}")
    work_root = config.root / "outputs/试听试验/.voca-work" / uuid.uuid4().hex[:10]
    work_root.mkdir(parents=True, exist_ok=True)
    ffmpeg = _which("ffmpeg")
    if not ffmpeg:
        raise FileNotFoundError("ffmpeg is required for scene audio assembly")
    concat_items: list[Path] = []
    transcript_lines: list[str] = []
    try:
        for index, line in enumerate(scene["lines"], 1):
            raw = work_root / f"line-{index:03}.wav"
            canonical = work_root / f"canonical-{index:03}.wav"
            synthesize(config, line["profile"], line["text"], raw)
            _canonicalize_audio(raw, canonical)
            concat_items.append(canonical)
            transcript_lines.append(line["text"].strip())
            pause_ms = int(line.get("pause_after_ms", 0))
            if pause_ms:
                pause = work_root / f"pause-{index:03}.wav"
                _run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                      "anullsrc=r=48000:cl=mono", "-t", f"{pause_ms / 1000:.3f}", "-c:a", "pcm_s16le", str(pause)])
                concat_items.append(pause)
        concat_file = work_root / "concat.txt"
        def quote_concat(path: Path) -> str:
            return "file '" + str(path).replace("'", "'\\''") + "'"
        concat_file.write_text("\n".join(quote_concat(path) for path in concat_items) + "\n", encoding="utf-8")
        audio_output.parent.mkdir(parents=True, exist_ok=True)
        _run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_file),
              "-c:a", "pcm_s16le", str(audio_output)])
        text_output.write_text("\n".join(transcript_lines) + "\n", encoding="utf-8")
        print(f"Scene audio: {audio_output}\nTranscript:  {text_output}")
        command_manifest(argparse.Namespace(media=str(audio_output), profile=None,
            parameters=json.dumps({"scene": title, "lines": scene["lines"]}, ensure_ascii=False), output=None), config)
        if args.image or args.background_video:
            video_args = argparse.Namespace(audio=str(audio_output), transcript_file=str(text_output), output=str(video_output),
                image=args.image, background_video=args.background_video, aligner_host=args.aligner_host,
                threads=args.threads, long_edge=args.long_edge, normalize_audio=args.normalize_audio,
                overwrite=args.overwrite)
            command_video(video_args, config)
    finally:
        shutil.rmtree(work_root, ignore_errors=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="voca", description="CPU-first voice production helpers")
    parser.add_argument("--config", help="TOML config (default: config/local.toml, then example config)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init-config", help="Create a local, git-ignored config")
    doctor = sub.add_parser("doctor", help="Show local tools and project configuration")
    doctor.add_argument("--remote", action="store_true", help="Also check SSH access and remote project root")
    env = sub.add_parser("env", help="Capture environment versions")
    env_sub = env.add_subparsers(dest="env_command", required=True)
    env_snapshot = env_sub.add_parser("snapshot", help="Save local and remote pip freeze metadata locally")
    env_snapshot.add_argument("--output")
    profiles = sub.add_parser("profiles", help="List named voice profiles")
    profiles.add_argument("name", nargs="?")
    synth = sub.add_parser("synth", help="Synthesize with a named profile on the CPU host")
    synth.add_argument("--profile", required=True)
    source = synth.add_mutually_exclusive_group(required=True)
    source.add_argument("--text")
    source.add_argument("--text-file")
    synth.add_argument("--output", required=True)
    synth.add_argument("--overwrite", action="store_true")
    for option, kind in (("index-rate", float), ("pitch", int), ("protect", float), ("rms-mix-rate", float),
                         ("top-k", int), ("top-p", float), ("temperature", float), ("speed", float),
                         ("target-rms-dbfs", float), ("output-gain", float)):
        synth.add_argument(f"--{option}", type=kind, default=None)
    manifest = sub.add_parser("manifest", help="Write a hash and media metadata sidecar")
    manifest.add_argument("media")
    manifest.add_argument("--profile")
    manifest.add_argument("--parameters", help="JSON object of synthesis settings")
    manifest.add_argument("--output")
    index = sub.add_parser("index-assets", help="Index local audio/video metadata to CSV")
    index.add_argument("--root")
    index.add_argument("--output")
    qc = sub.add_parser("qc", help="Inspect media streams, duration, sample rate and resolution")
    qc.add_argument("media")
    compare = sub.add_parser("compare", help="Create a local HTML A/B listening page")
    compare.add_argument("media", nargs="+", help="Audio files to compare")
    compare.add_argument("--output")
    archive = sub.add_parser("archive", help="Dry-run or move trial media into the dated archive")
    archive.add_argument("path")
    archive.add_argument("--apply", action="store_true")
    sweep = sub.add_parser("sweep", help="Run selectable Qwen-to-RVC CPU parameter sweep remotely")
    sweep.add_argument("--variants", help="Comma-separated labels, e.g. i015_p0,i030_p-1")
    sweep.add_argument("--source-only", action="store_true")
    sweep.add_argument("--skip-source", action="store_true")
    sweep.add_argument("--dry-run", action="store_true")
    sweep.add_argument("--overwrite", action="store_true")
    sweep.add_argument("--output-gain", type=float, default=None, help="Linear gain multiplier (default from [audio])")
    sweep.add_argument("--peak-limit-dbfs", type=float, default=None,
                       help="Peak ceiling in dBFS (default from [audio])")
    video = sub.add_parser("video", help="Align subtitles and render video from a still or video background")
    bg = video.add_mutually_exclusive_group(required=True)
    bg.add_argument("--image")
    bg.add_argument("--background-video")
    video.add_argument("--audio", required=True)
    video.add_argument("--transcript-file", required=True)
    video.add_argument("--output", required=True)
    video.add_argument("--aligner-host")
    video.add_argument("--threads", type=int, default=8)
    video.add_argument("--long-edge", type=int, default=2048)
    video.add_argument("--normalize-audio", action="store_true")
    video.add_argument("--overwrite", action="store_true")
    scene = sub.add_parser("scene", help="Validate or render a JSON short-theater script")
    scene_sub = scene.add_subparsers(dest="scene_command", required=True)
    scene_validate = scene_sub.add_parser("validate")
    scene_validate.add_argument("scene")
    scene_render = scene_sub.add_parser("render")
    scene_render.add_argument("scene")
    scene_render.add_argument("--output")
    scene_render.add_argument("--video-output")
    scene_bg = scene_render.add_mutually_exclusive_group()
    scene_bg.add_argument("--image")
    scene_bg.add_argument("--background-video")
    scene_render.add_argument("--aligner-host")
    scene_render.add_argument("--threads", type=int, default=8)
    scene_render.add_argument("--long-edge", type=int, default=2048)
    scene_render.add_argument("--normalize-audio", action="store_true")
    scene_render.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        if args.command == "init-config":
            command_init_config(config)
        elif args.command == "doctor":
            command_doctor(config, args.remote)
        elif args.command == "env" and args.env_command == "snapshot":
            command_env_snapshot(args, config)
        elif args.command == "profiles":
            command_profiles(config, args.name)
        elif args.command == "synth":
            text = args.text if args.text is not None else Path(args.text_file).expanduser().read_text(encoding="utf-8-sig")
            overrides = {key.replace("-", "_"): getattr(args, key.replace("-", "_")) for key in
                         ("index-rate", "pitch", "protect", "rms-mix-rate", "top-k", "top-p", "temperature", "speed", "target-rms-dbfs", "output-gain")}
            selected_profile = dict(_profile(config, args.profile))
            selected_profile.update({key: value for key, value in overrides.items() if value is not None})
            selected_profile.setdefault("output_gain", float(config.values.get("audio", {}).get("output_gain", DEFAULT_OUTPUT_GAIN)))
            synthesize(config, args.profile, text, Path(args.output), overrides, args.overwrite)
            command_manifest(argparse.Namespace(media=args.output, profile=args.profile,
                parameters=json.dumps(selected_profile, ensure_ascii=False), output=None), config)
        elif args.command == "manifest":
            command_manifest(args, config)
        elif args.command == "index-assets":
            command_index(args, config)
        elif args.command == "qc":
            command_qc(args)
        elif args.command == "compare":
            command_compare(args, config)
        elif args.command == "archive":
            command_archive(args, config)
        elif args.command == "sweep":
            command_sweep(args, config)
        elif args.command == "video":
            command_video(args, config)
        elif args.command == "scene" and args.scene_command == "validate":
            command_scene_validate(args, config)
        elif args.command == "scene" and args.scene_command == "render":
            command_scene_render(args, config)
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.CalledProcessError, json.JSONDecodeError) as error:
        print(f"voca: error: {error}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
