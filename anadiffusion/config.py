"""JSON configuration loading with placeholder-path validation."""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, Tuple

PLACEHOLDER_PREFIX = "<PATH-TO-"


def parse_ints(value: str) -> Tuple[int, ...]:
    return tuple(int(v.strip()) for v in str(value).split(","))


def parse_floats(value: str) -> Tuple[float, ...]:
    return tuple(float(v.strip()) for v in str(value).split(","))


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_parser(description: str, defaults: Dict[str, Any]) -> argparse.ArgumentParser:
    """Parser whose options are the config keys; command-line values override the JSON."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="JSON config file.")
    for key, value in defaults.items():
        if isinstance(value, bool):
            parser.add_argument(f"--{key}", type=parse_bool, default=None)
        else:
            parser.add_argument(f"--{key}", type=type(value) if value is not None else str, default=None)
    return parser


def resolve(parser: argparse.ArgumentParser, defaults: Dict[str, Any]) -> argparse.Namespace:
    """Merge defaults < config file < command line, and reject placeholder paths."""
    known = parser.parse_args()
    config = dict(defaults)
    config.update(load_config(known.config))
    for key, value in vars(known).items():
        if key != "config" and value is not None:
            config[key] = value
    unknown = sorted(set(config) - set(defaults))
    if unknown:
        parser.error(f"Unknown config key(s): {', '.join(unknown)}")
    placeholders = [k for k, v in config.items() if isinstance(v, str) and v.startswith(PLACEHOLDER_PREFIX)]
    if placeholders:
        parser.error(f"Replace placeholder path(s) in the config before running: {', '.join(placeholders)}")
    return argparse.Namespace(**config)
