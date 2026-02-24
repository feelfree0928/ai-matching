#!/usr/bin/env python3
"""
Matching quality evaluation script.

Runs a set of test job descriptions against the live index and produces a
human-readable quality report with:
  - Score breakdown per result
  - Title relevance flags
  - Rank inversion detection (bad result ranked above good one)
  - NDCG@5 approximation (based on title keyword match)
  - Pass / Fail per test case

Usage:
    python scripts/eval_matching.py                          # uses tests/eval_cases.json
    python scripts/eval_matching.py --cases path/to/file.json
    python scripts/eval_matching.py --api http://host:8000   # run against remote API
    python scripts/eval_matching.py --top 5                  # show top-N results per case
    python scripts/eval_matching.py --save report.json       # save machine-readable report
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Any

# ── Allow running from project root or scripts/ ─────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

CASES_PATH = os.path.join(os.path.dirname(__file__), "..", "tests", "eval_cases.json")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _kw_hit(text: str, keywords: list[str]) -> bool:
    """True if any keyword appears in text (case-insensitive)."""
    if not text or not keywords:
        return False
    t = text.lower()
    return any(kw.lower() in t for kw in keywords)


def _title_relevance(candidate_role: str, title_keywords: list[str]) -> int:
    """0 = no match, 1 = partial match, 2 = strong match."""
    if not candidate_role or not title_keywords:
        return 0
    role = candidate_role.lower()
    hits = sum(1 for kw in title_keywords if kw.lower() in role)
    if hits == 0:
        return 0
    if hits >= 2:
        return 2
    return 1


def _ndcg_at_k(relevance_scores: list[int], k: int) -> float:
    """Simple NDCG@k where relevance_scores[i] = relevance of rank i+1."""
    def _dcg(rels: list[int]) -> float:
        return sum(
            (2 ** r - 1) / math.log2(i + 2)
            for i, r in enumerate(rels[:k])
        )
    ideal = sorted(relevance_scores, reverse=True)
    dcg = _dcg(relevance_scores)
    idcg = _dcg(ideal)
    return dcg / idcg if idcg > 0 else 0.0


def _color(text: str, code: str) -> str:
    """ANSI colour wrap."""
    return f"\033[{code}m{text}\033[0m"


def _green(t: str) -> str: return _color(t, "32")
def _red(t: str) -> str:   return _color(t, "31")
def _yellow(t: str) -> str: return _color(t, "33")
def _bold(t: str) -> str:  return _color(t, "1")
def _dim(t: str) -> str:   return _color(t, "2")


# ── API calls ─────────────────────────────────────────────────────────────────

def call_api(base_url: str, case: dict) -> tuple[list[dict], float]:
    """Call POST /api/match. Returns (matches, latency_ms)."""
    import urllib.request

    payload = {k: v for k, v in case.items() if k != "expect" and k != "id"}
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/match",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read())
    except Exception as e:
        return [], 0.0
    latency = (time.perf_counter() - t0) * 1000
    return body.get("matches", []), latency


def call_local(case: dict) -> tuple[list[dict], float]:
    """Call matching directly without HTTP (requires local ES)."""
    from api.matching import run_match
    from api.models import JobMatchRequest

    payload = {k: v for k, v in case.items() if k != "expect" and k != "id"}
    req = JobMatchRequest(**payload)
    t0 = time.perf_counter()
    resp = run_match(req)
    latency = (time.perf_counter() - t0) * 1000
    matches = [m.model_dump() for m in resp.matches]
    return matches, latency


# ── Per-result analysis ──────────────────────────────────────────────────────

def analyse_result(match: dict, title_keywords: list[str], industry_keywords: list[str]) -> dict:
    """Return flags dict for one result."""
    role = (match.get("most_relevant_role") or match.get("candidate_name") or "").strip()
    industries = match.get("top_industries") or match.get("most_experience_industries") or []
    score = match.get("score", {})

    title_rel = _title_relevance(role, title_keywords)
    ind_hit = any(_kw_hit(ind, industry_keywords) for ind in industries) if industry_keywords else None

    flags = []
    if title_rel == 0:
        flags.append("⚠ title_mismatch")
    if ind_hit is False:
        flags.append("⚠ industry_mismatch")
    if score.get("title_score", 0) < 5:
        flags.append("⚠ low_title_score")

    return {
        "role": role,
        "title_relevance": title_rel,
        "industry_hit": ind_hit,
        "score_total": score.get("total", 0),
        "score_breakdown": score,
        "flags": flags,
    }


# ── Report formatting ────────────────────────────────────────────────────────

SCORE_BAR_WIDTH = 20

def _score_bar(score: float, max_score: float = 100) -> str:
    filled = int(SCORE_BAR_WIDTH * score / max(max_score, 1))
    bar = "█" * filled + "░" * (SCORE_BAR_WIDTH - filled)
    return f"[{bar}] {score:.1f}"


def _relevance_icon(rel: int) -> str:
    return ["✗", "~", "✓"][rel]


def print_case_report(case: dict, matches: list[dict], latency: float, top_n: int) -> dict:
    expect = case.get("expect", {})
    title_keywords = expect.get("top1_title_keywords", [])
    industry_keywords = expect.get("top3_industry_keywords", [])
    min_top1_score = expect.get("min_top1_score", 0)
    max_rank_inversion = expect.get("max_rank_inversion", 99)

    analysed = [analyse_result(m, title_keywords, industry_keywords) for m in matches[:top_n]]

    # ── Quality metrics ─────────────────────────────────────────────────────
    rel_scores = [a["title_relevance"] for a in analysed]
    ndcg5 = _ndcg_at_k(rel_scores, 5)

    rank_inversions = 0
    for i in range(len(analysed) - 1):
        if analysed[i]["title_relevance"] < analysed[i + 1]["title_relevance"]:
            rank_inversions += 1

    top1_ok = len(analysed) > 0 and analysed[0]["title_relevance"] > 0
    top1_score_ok = len(analysed) > 0 and analysed[0]["score_total"] >= min_top1_score
    inversions_ok = rank_inversions <= max_rank_inversion

    passed = top1_ok and top1_score_ok and inversions_ok

    # ── Header ───────────────────────────────────────────────────────────────
    status = _green("PASS ✓") if passed else _red("FAIL ✗")
    print()
    print(_bold(f"{'─'*70}"))
    print(_bold(f" [{case['id']}]  {case['title']}    {status}    {latency:.0f} ms"))
    print(f"{'─'*70}")
    print(f"  NDCG@5 = {ndcg5:.2f}   |   rank inversions = {rank_inversions}   |   total returned = {len(matches)}")
    print()

    # ── Per-result table ─────────────────────────────────────────────────────
    col_w = 35
    print(f"  {'Rank':<5}{'Relevance':<10}{'Score':>22}  {'Role / Title'}")
    print(f"  {'─'*4} {'─'*8} {'─'*22}  {'─'*(col_w)}")

    for i, (m, a) in enumerate(zip(matches[:top_n], analysed)):
        rank_str = f"#{i+1}"
        rel_icon = _relevance_icon(a["title_relevance"])
        rel_colored = (
            _green(f"  {rel_icon} good   ")
            if a["title_relevance"] == 2 else (
                _yellow(f"  {rel_icon} partial")
                if a["title_relevance"] == 1 else
                _red(f"  {rel_icon} MISMATCH")
            )
        )
        score_bar = _score_bar(a["score_total"])
        name_str = (m.get("candidate_name") or "").strip()
        role_str = a["role"] or _dim("(no role)")
        display = f"{role_str}" + (f"  [{name_str}]" if name_str else "")
        if len(display) > col_w + 10:
            display = display[: col_w + 7] + "..."

        inversion_flag = _red(" ← RANK INVERSION") if (
            i > 0 and analysed[i]["title_relevance"] > analysed[i - 1]["title_relevance"]
        ) else ""

        extra_flags = "  " + "  ".join(a["flags"]) if a["flags"] else ""

        print(f"  {rank_str:<5}{rel_colored}  {score_bar}  {display}{inversion_flag}{extra_flags}")

        # Score breakdown (compact)
        bd = a["score_breakdown"]
        breakdown = (
            f"    title={bd.get('title_score',0):.1f}"
            f"  industry={bd.get('industry_score',0):.1f}"
            f"  exp={bd.get('experience_score',0):.1f}"
            f"  skills={bd.get('skills_score',0):.1f}"
            f"  seniority={bd.get('seniority_score',0):.1f}"
            f"  edu={bd.get('education_score',0):.1f}"
            f"  lang={bd.get('language_score',0):.1f}"
        )
        print(_dim(breakdown))

    # ── Failure details ──────────────────────────────────────────────────────
    print()
    issues = []
    if not top1_ok:
        issues.append(_red(f"  ✗ Top result title doesn't match any keyword: {title_keywords}"))
    if not top1_score_ok:
        actual = analysed[0]["score_total"] if analysed else 0
        issues.append(_red(f"  ✗ Top result score {actual:.1f} < expected minimum {min_top1_score}"))
    if not inversions_ok:
        issues.append(_red(f"  ✗ {rank_inversions} rank inversions (max allowed: {max_rank_inversion})"))
    for issue in issues:
        print(issue)

    return {
        "id": case["id"],
        "title": case["title"],
        "passed": passed,
        "ndcg5": round(ndcg5, 3),
        "rank_inversions": rank_inversions,
        "latency_ms": round(latency),
        "top1_score": analysed[0]["score_total"] if analysed else 0,
        "top1_title_relevance": analysed[0]["title_relevance"] if analysed else 0,
        "results_count": len(matches),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate matching quality against the live index")
    parser.add_argument("--cases", default=CASES_PATH, help="Path to eval_cases.json")
    parser.add_argument("--api", default=None,
                        help="Backend base URL (e.g. http://74.161.162.184:8000). "
                             "If omitted, calls matching directly (local ES required).")
    parser.add_argument("--top", type=int, default=5, help="Number of results to show per case (default: 5)")
    parser.add_argument("--save", default=None, help="Save JSON report to this path")
    parser.add_argument("--case", default=None, help="Run only this case ID (e.g. ACC-001)")
    args = parser.parse_args()

    with open(args.cases, "r", encoding="utf-8") as f:
        cases = json.load(f)

    if args.case:
        cases = [c for c in cases if c["id"] == args.case]
        if not cases:
            print(f"Case '{args.case}' not found.")
            sys.exit(1)

    print(_bold(f"\n{'='*70}"))
    print(_bold(f"  MATCHING QUALITY EVALUATION  —  {len(cases)} test case(s)"))
    if args.api:
        print(f"  Backend: {args.api}")
    else:
        print("  Mode: direct (local Elasticsearch)")
    print(_bold(f"{'='*70}"))

    reports = []
    for case in cases:
        if args.api:
            matches, latency = call_api(args.api, case)
        else:
            matches, latency = call_local(case)

        report = print_case_report(case, matches, latency, top_n=args.top)
        reports.append(report)

    # ── Summary table ─────────────────────────────────────────────────────
    print()
    print(_bold(f"\n{'='*70}"))
    print(_bold("  SUMMARY"))
    print(f"{'='*70}")
    total = len(reports)
    passed = sum(1 for r in reports if r["passed"])
    avg_ndcg = sum(r["ndcg5"] for r in reports) / total if total else 0
    avg_latency = sum(r["latency_ms"] for r in reports) / total if total else 0

    print(f"  Pass rate:    {passed}/{total}  ({100*passed//total if total else 0}%)")
    print(f"  Avg NDCG@5:   {avg_ndcg:.3f}  (1.0 = perfect, 0.5 = mediocre)")
    print(f"  Avg latency:  {avg_latency:.0f} ms")
    print()
    print(f"  {'ID':<12}{'Pass':<7}{'NDCG@5':<10}{'Top1 Score':<12}{'Inversions':<12}{'ms'}")
    print(f"  {'─'*11} {'─'*6} {'─'*9} {'─'*11} {'─'*11} {'─'*6}")
    for r in reports:
        pstr = _green("PASS") if r["passed"] else _red("FAIL")
        ndcg_str = _green(f"{r['ndcg5']:.3f}") if r["ndcg5"] >= 0.7 else (
            _yellow(f"{r['ndcg5']:.3f}") if r["ndcg5"] >= 0.4 else _red(f"{r['ndcg5']:.3f}")
        )
        print(f"  {r['id']:<12}{pstr:<15}{ndcg_str:<18}{r['top1_score']:<12.1f}{r['rank_inversions']:<12}{r['latency_ms']}")

    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump({
                "summary": {
                    "pass_rate": f"{passed}/{total}",
                    "avg_ndcg5": round(avg_ndcg, 3),
                    "avg_latency_ms": round(avg_latency),
                },
                "cases": reports,
            }, f, indent=2)
        print(f"\n  Report saved to: {args.save}")

    print()
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
