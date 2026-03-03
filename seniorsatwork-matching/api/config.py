"""
Runtime-configurable scoring weights and threshold. Persisted to config.json.
"""
from __future__ import annotations

import json
import os

CONFIG_PATH = os.getenv("CONFIG_PATH", os.path.join(os.path.dirname(__file__), "..", "config.json"))

DEFAULT_WEIGHTS = {
    "title":      0.23,   # reduced so experience + industry can overtake title-only similarity
    "industry":   0.17,   # raised so Financial vs Education/Automotive clearly affects ranking (Senior Accountant)
    "experience": 0.25,   # many-yr relevant role (e.g. Accountant) can beat short exact-title match
    "skills":     0.20,
    "seniority":  0.07,
    "education":  0.05,
    "language":   0.00,
}

# Raw score threshold (on script_score scale ~0..2). 1.45 ≈ 72.5/100 normalized.
# Lowered to compensate for reduced title ceiling (title weight 0.48→0.35 lowers raw scores).
# Acts as an upper cap: stored config values higher than this are ignored automatically.
DEFAULT_MIN_SCORE_RAW = 1.45
DEFAULT_MAX_RESULTS = 5


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
    # Cap stored value at DEFAULT_MIN_SCORE_RAW so code-side threshold reductions
    # take effect immediately even when an older config.json has a higher value.
    stored = float(load_config().get("min_score_raw", DEFAULT_MIN_SCORE_RAW))
    return min(stored, DEFAULT_MIN_SCORE_RAW)


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
