# Project-voca

CPU-first voice production workspace for named voice profiles, audio conversion, small-theater scenes, subtitle alignment, and short-video rendering.

## Quick start

```bash
python3 -m pip install -e '.[video]'
voca init-config
voca doctor
voca profiles
```

Edit the ignored `config/local.toml` to set the remote CPU host and project path. The model environments stay isolated on that host. See [Reusable workflows](docs/可复用工作流与下一步.md) for the full command guide and [baseline recipes](docs/若叶睦基线复刻指南.md) for the four named voice profiles.

## Current voice references

- **Quiet Mutsumi**: Qwen source speech followed by RVC at `index=0.15`, `pitch=0`, `protect=0.45`.
- **Daily Mutsumi**: Qwen source speech followed by RVC at `index=0.30`, `pitch=-1`, `protect=0.50`.
- **Sobbing Mutsumi v1**: short-data GPT-SoVITS tuning with a local five-second reference clip.
- **Excited Mutsumi**: the selected GSV output itself. It stays as unpolished GSV audio; no automatic RVC pass or loudness-matched derivative replaces it.

The confirmed WAVs, models, training/reference audio, dialogue text, and video assets remain in the local workspace and are excluded from this public repository. The repo tracks reusable code, profile settings, and operating notes. A local checkout can open the corresponding baseline files under `outputs/基线/`.

## Project map

- `voca/`: unified CLI for synthesis dispatch, profiles, manifests, QC, indexing, comparison, archive, and scenes.
- `scripts/`: Qwen/GSV/RVC CPU wrappers, text preparation, F0 inspection, forced alignment, and video rendering.
- `configs/`: example project settings, named profiles, and a generic scene template.
- `docs/`: setup, reproducible baselines, model/environment notes, and workflow improvements.
- `outputs/`, `videos/`, `training_data/`, `models/`, `frameworks/`, `assets/`: local working data; ignored by Git where appropriate.

All model synthesis recipes explicitly hide CUDA. See [documentation index](docs/README.md) for focused references.
