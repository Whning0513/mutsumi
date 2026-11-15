from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.9/3.10
    import tomli as tomllib


@dataclass(frozen=True)
class ProjectConfig:
    file: Path
    root: Path
    values: dict[str, Any]

    @property
    def remote(self) -> dict[str, Any]:
        return self.values.get("remote", {})

    @property
    def runtime(self) -> dict[str, Any]:
        return self.values.get("runtime", {})

    def local_path(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (self.root / path).resolve()

    def remote_path(self, value: str | Path) -> str:
        from posixpath import join

        path = str(value)
        if path.startswith("/"):
            return path
        return join(str(self.remote["root"]).rstrip("/"), path)


def load_config(path: str | Path | None = None) -> ProjectConfig:
    if path:
        config_file = Path(path).expanduser().resolve()
    else:
        cwd = Path.cwd()
        candidates = [cwd / "config/local.toml", cwd / "configs/project.example.toml"]
        config_file = next((item for item in candidates if item.is_file()), candidates[-1])
    if not config_file.is_file():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    values = tomllib.loads(config_file.read_text(encoding="utf-8"))
    local_root = Path(values.get("project", {}).get("local_root", ".")).expanduser()
    root = (config_file.parent / local_root).resolve() if not local_root.is_absolute() else local_root.resolve()
    return ProjectConfig(config_file, root, values)


def load_profiles(config: ProjectConfig) -> dict[str, dict[str, Any]]:
    filename = config.values.get("project", {}).get("profile_file", "configs/voice_profiles.toml")
    path = config.local_path(filename)
    profiles = tomllib.loads(path.read_text(encoding="utf-8")).get("profiles", {})
    if not profiles:
        raise ValueError(f"No voice profiles defined in {path}")
    return profiles
