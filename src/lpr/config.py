# OWNER: SHARED
"""Đọc cấu hình YAML và truy cập khóa dạng chấm (xem SPEC mục 7)."""
from __future__ import annotations

import copy
from typing import Any

import yaml


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Merge sâu `overrides` vào bản sao của `base`, trả về dict mới."""
    out = copy.deepcopy(base)
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str = "config/default.yaml", overrides: dict | None = None) -> dict:
    """Đọc file YAML cấu hình.

    Input: đường dẫn file YAML, dict `overrides` (có thể lồng nhau) để ghi đè.
    Output: dict cấu hình đã merge sâu.
    """
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if overrides:
        cfg = _deep_merge(cfg, overrides)
    return cfg


def get(cfg: dict, dotted_key: str, default: Any = None) -> Any:
    """Lấy giá trị theo khóa dạng chấm, ví dụ `get(cfg, "localize.top_k")`.

    Trả về `default` nếu một khóa trung gian không tồn tại.
    """
    cur: Any = cfg
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur
