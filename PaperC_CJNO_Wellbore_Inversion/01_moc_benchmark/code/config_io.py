# -*- coding: utf-8 -*-
"""Load / validate the YAML configuration files in ``configs/``."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import yaml

from paths import CONFIG_DIR, SCHEMA_DIR


def _coerce_numeric_strings(obj):
    """PyYAML 1.1 treats 4.0e7 (no exponent sign) as a string.  Coerce those."""
    if isinstance(obj, dict):
        return {k: _coerce_numeric_strings(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_coerce_numeric_strings(v) for v in obj]
    if isinstance(obj, str):
        s = obj.strip()
        if s and s[0] in "0123456789.+-" and any(c in s for c in "eE") and " " not in s:
            try:
                return float(s)
            except ValueError:
                return obj
    return obj


def load_yaml(name: str) -> Dict:
    with open(CONFIG_DIR / name, "r", encoding="utf-8") as f:
        return _coerce_numeric_strings(yaml.safe_load(f))


def load_all() -> Dict[str, Dict]:
    return {
        "physics": load_yaml("physics.yaml"),
        "friction_zielke": load_yaml("friction_zielke.yaml"),
        "friction_brunone": load_yaml("friction_brunone.yaml"),
        "data": load_yaml("data.yaml"),
    }


def load_schema(name: str) -> Dict:
    with open(SCHEMA_DIR / name, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_against_schema(obj: Dict, schema_name: str) -> None:
    import jsonschema
    jsonschema.validate(obj, load_schema(schema_name))


def save_yaml(obj: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False)
