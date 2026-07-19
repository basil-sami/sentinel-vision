from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None


DEFAULTS_DIR = Path(__file__).resolve().parent.parent / "configs" / "defaults"


def _load_yaml(path: str | Path) -> dict:
    if yaml is None:
        raise ImportError("PyYAML required: pip install pyyaml")
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


class Config:
    def __init__(self, data: dict):
        self._data = data

    def get(self, key: str, default: Any = None) -> Any:
        parts = key.split(".")
        val = self._data
        for p in parts:
            if isinstance(val, dict):
                val = val.get(p)
            else:
                return default
            if val is None:
                return default
        return val

    @property
    def raw(self) -> dict:
        return self._data

    def as_dict(self) -> dict:
        return self._data.copy()


class ConfigLoader:
    def __init__(self, config_dir: str | Path | None = None):
        self._defaults = {}
        self._user_configs = {}
        if config_dir:
            self._load_user_configs(config_dir)

    def _load_defaults(self):
        if self._defaults:
            return
        for yaml_file in DEFAULTS_DIR.glob("*.yaml"):
            key = yaml_file.stem
            self._defaults[key] = _load_yaml(yaml_file)

    def _load_user_configs(self, config_dir: str | Path):
        config_dir = Path(config_dir)
        if not config_dir.exists():
            return
        for yaml_file in config_dir.glob("*.yaml"):
            key = yaml_file.stem
            self._user_configs[key] = _load_yaml(yaml_file)

    def load(self, section: str) -> Config:
        self._load_defaults()
        base = self._defaults.get(section, {})
        override = self._user_configs.get(section, {})
        merged = _deep_merge(base, override)
        return Config(merged)

    def all(self) -> dict[str, Config]:
        self._load_defaults()
        result = {}
        all_keys = set(self._defaults.keys()) | set(self._user_configs.keys())
        for key in all_keys:
            base = self._defaults.get(key, {})
            override = self._user_configs.get(key, {})
            result[key] = Config(_deep_merge(base, override))
        return result
