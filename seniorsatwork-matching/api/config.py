"""
Runtime-configurable scoring weights and threshold. Persisted to config.json.
"""
from __future__ import annotations

import json
import os

CONFIG_PATH = os.getenv("CONFIG_PATH", os.path.join(os.path.dirname(__file__), "..", "config.json"))

DEFAULT_WEIGHTS = {
    "title": 0.40,
    "industry": 0.20,
    "experience": 0.15,
    "skills": 0.10,
    "seniority": 0.08,
    "education": 0.07,
}

# Raw score threshold (on script_score scale ~0..2). 1.55 ≈ 55/100 normalized.
DEFAULT_MIN_SCORE_RAW = 1.55
DEFAULT_MAX_RESULTS = 20


def load_config() -> dict:
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "scoring_weights": DEFAULT_WEIGHTS.copy(),
        "min_score_raw": DEFAULT_MIN_SCORE_RAW,
        "max_results": DEFAULT_MAX_RESULTS,
    }


def save_config(cfg: dict) -> None:
    os.makedirs(os.path.dirname(CONFIG_PATH) or ".", exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def get_weights() -> dict[str, float]:
    return load_config().get("scoring_weights", DEFAULT_WEIGHTS).copy()


def get_min_score_raw() -> float:
    return float(load_config().get("min_score_raw", DEFAULT_MIN_SCORE_RAW))


def get_max_results() -> int:
    return int(load_config().get("max_results", DEFAULT_MAX_RESULTS))


def update_config(updates: dict) -> dict:
    cfg = load_config()
    if "scoring_weights" in updates:
        cfg["scoring_weights"] = {**cfg.get("scoring_weights", {}), **updates["scoring_weights"]}
    if "min_score_raw" in updates:
        cfg["min_score_raw"] = updates["min_score_raw"]
    if "max_results" in updates:
        cfg["max_results"] = updates["max_results"]
    save_config(cfg)
    return cfg
