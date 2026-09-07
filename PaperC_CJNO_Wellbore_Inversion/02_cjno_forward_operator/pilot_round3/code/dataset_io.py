# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from paths import ROUND1_MANIFEST


def load_round1_manifest(path: Optional[Path] = None) -> dict:
    p = path or ROUND1_MANIFEST
    return json.loads(p.read_text(encoding="utf-8"))
