"""
Runtime-configurable scoring weights and threshold. Persisted to config.json.
"""
from __future__ import annotations

import json
import os

CONFIG_PATH = os.getenv("CONFIG_PATH", os.path.join(os.path.dirname(__file__), "..", "config.json"))

DEFAULT_WEIGHTS = {
    "title":      0.15,   # role alignment; typical industry balance
    "industry":   0.12,   # sector fit
    "experience": 0.25,   # years + relevance
    "skills":     0.35,   # primary signal (highest weight)
    "seniority":  0.06,   # level match
    "education":  0.05,   # degree/certifications
    "language":   0.02,   # minimal
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


def get_max_raw_score(weights: dict[str, float] | None = None) -> float:
    """
    Theoretical maximum raw score when every dimension is at its maximum.
    Used so display score = (raw_score / max_raw_score) × 100 (probability, never exceeds 100).
    Dimension maxes: title/industry/skills/edu/seniority/language = 2.0 each; experience = 2.6.
    """
    w = weights if weights is not None else get_weights()
    w_t = w.get("title", DEFAULT_WEIGHTS["title"])
    w_i = w.get("industry", DEFAULT_WEIGHTS["industry"])
    w_e = w.get("experience", DEFAULT_WEIGHTS["experience"])
    w_s = w.get("skills", DEFAULT_WEIGHTS["skills"])
    w_sen = w.get("seniority", DEFAULT_WEIGHTS["seniority"])
    w_edu = w.get("education", DEFAULT_WEIGHTS["education"])
    w_lang = w.get("language", DEFAULT_WEIGHTS["language"])
    return 2.0 * (w_t + w_i + w_s + w_sen + w_edu + w_lang) + 2.6 * w_e


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
