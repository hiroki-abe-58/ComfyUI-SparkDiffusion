"""Runtime settings resolution: node input > environment variable > local config file.

Keeping machine-specific paths out of workflows means a workflow JSON can be
shared without leaking (or depending on) anyone's directory layout.

Local config file (git-ignored): ``sparkdiffusion_config.json`` next to this
package's ``__init__.py``, or the file named by ``SPARKDIFFUSION_CONFIG``.
See ``sparkdiffusion_config.example.json``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILENAME = "sparkdiffusion_config.json"

ENV_KEYS = {
    "repo": "SPARKDIFFUSION_REPO",
    "python": "SPARKDIFFUSION_PYTHON",
    "model_root": "SPARKDIFFUSION_MODEL_ROOT",
    "backend": "SPARKDIFFUSION_BACKEND",
    "wsl_distro": "SPARKDIFFUSION_WSL_DISTRO",
    "sla_src": "SPARKDIFFUSION_SLA_SRC",
}


def config_file_path() -> Path:
    override = os.environ.get("SPARKDIFFUSION_CONFIG")
    return Path(override) if override else PACKAGE_ROOT / CONFIG_FILENAME


def load_config_file(path: Optional[Path] = None) -> Dict[str, Any]:
    path = path or config_file_path()
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read SparkDiffusion config file {path.name}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"SparkDiffusion config file {path.name} must contain a JSON object")
    return data


def resolve(
    key: str,
    node_value: Optional[str],
    file_cfg: Optional[Dict[str, Any]] = None,
    environ: Optional[Dict[str, str]] = None,
) -> str:
    """First non-empty of: node input, environment variable, config file."""
    if node_value is not None and str(node_value).strip():
        return str(node_value).strip()
    env = os.environ if environ is None else environ
    env_key = ENV_KEYS.get(key)
    if env_key and env.get(env_key, "").strip():
        return env[env_key].strip()
    cfg = load_config_file() if file_cfg is None else file_cfg
    value = cfg.get(key)
    return str(value).strip() if value is not None else ""
