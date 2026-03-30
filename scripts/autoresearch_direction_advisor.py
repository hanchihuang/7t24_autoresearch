#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from autoresearch_helpers import (
    AutoresearchError,
    normalize_labels,
    parse_results_log,
    read_state_payload,
    resolve_repo_path,
    resolve_repo_relative,
    resolve_state_path_for_log,
)


GENERIC_LABELS = {
    "continuation_path",
    "confirm200",
    "full_confirmation",
    "strict_xml_recovered",
    "scout_gate_pass",
    "retained_control",
    "scout_protocol",
    "web_search",
}

FAMILY_RULES: list[tuple[tuple[str, ...], list[str], str]] = [
    (
        ("synthetic", "template", "synth", "schema"),
        [
            "reward_side_shaping",
            "slice_aware_shaping",
            "verifier_guided_reward",
        ],
        "Recent failures cluster around synthetic/template variants, so the next gain is more likely to come from reward/value redesign than narrower data mixing.",
    ),
    (
        ("replay", "buffer", "difficulty_prompt"),
        [
            "reward_side_shaping",
            "verifier_rerank_or_penalty",
            "hard_slice_curriculum",
        ],
        "Replay-style attempts are not converting into stable metric gains; switch from prompt reuse mechanics to stronger value signals or failure-slice shaping.",
    ),
    (
        ("verifier",),
        [
            "verifier_integration_hardening",
            "pairwise_data_quality",
            "conservative_mainline_integration",
        ],
        "Verifier-related work is active; the best next move is usually to improve signal quality or integrate it more conservatively into the mainline.",
    ),
    (
        ("reward", "step_alignment"),
        [
            "reward_path_activation_checks",
            "slice_aware_reward",
            "intermediate_step_reward",
        ],
        "Reward-side hypotheses are present but may not be engaging cleanly; move toward explicit activation checks and slice-specific reward design.",
    ),
]


def infer_family(label: str) -> str:
    text = label.strip().lower()
    if not text or text in GENERIC_LABELS:
        return ""
    for marker, _, _ in FAMILY_RULES:
        if any(token in text for token in marker):
            return marker[0]
    return text


def derive_exhausted_families(recent_labels: list[str]) -> list[str]:
    families = [infer_family(label) for label in recent_labels]
    families = [family for family in families if family]
    counts = Counter(families)
    return [family for family, count in counts.items() if count >= 2]


def suggested_directions_for_labels(recent_labels: list[str]) -> tuple[list[str], list[str]]:
    suggestions: list[str] = []
    rationales: list[str] = []
    lowered = [label.lower() for label in recent_labels]
    for markers, family_suggestions, rationale in FAMILY_RULES:
        if any(any(marker in label for marker in markers) for label in lowered):
            suggestions.extend(family_suggestions)
            rationales.append(rationale)
    if not suggestions:
        suggestions.extend(
            [
                "reward_side_shaping",
                "slice_aware_shaping",
                "verifier_guided_selection",
                "broader_hypothesis_family",
            ]
        )
        rationales.append(
            "No single exhausted family dominates the recent history; broaden the search to a different hypothesis family instead of local parameter churn."
        )
    deduped = list(dict.fromkeys(suggestions))
    deduped_rationales = list(dict.fromkeys(rationales))
    return deduped, deduped_rationales


def evaluate_direction_shift(
    *,
    results_path: Path,
    state_path: Path,
    recent_window: int = 8,
) -> dict[str, Any]:
    parsed = parse_results_log(results_path)
    payload = read_state_payload(state_path)
    main_rows = parsed.main_rows
    recent_rows = main_rows[-recent_window:] if recent_window > 0 else main_rows
    recent_statuses = [row.status for row in recent_rows]
    recent_labels = [label for row in recent_rows for label in normalize_labels(row.labels)]
    exhausted_families = derive_exhausted_families(recent_labels)
    suggested_directions, rationales = suggested_directions_for_labels(recent_labels)

    state = payload.get("state", {})
    consecutive_discards = int(state.get("consecutive_discards", 0) or 0)
    pivot_count = int(state.get("pivot_count", 0) or 0)
    should_shift = consecutive_discards >= 4 or pivot_count >= 2 or len(exhausted_families) >= 1
    shift_reason = ""
    if consecutive_discards >= 4:
      shift_reason = "consecutive_discards"
    elif pivot_count >= 2:
      shift_reason = "pivot_accumulation"
    elif exhausted_families:
      shift_reason = "family_exhausted"

    return {
        "should_shift_direction": should_shift,
        "shift_reason": shift_reason,
        "recent_window": len(recent_rows),
        "recent_statuses": recent_statuses,
        "recent_labels": recent_labels,
        "exhausted_families": exhausted_families,
        "suggested_directions": suggested_directions,
        "rationales": rationales,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Suggest whether an autoresearch run should switch to a new hypothesis family."
    )
    parser.add_argument("--repo", help="Primary repo root.")
    parser.add_argument("--results-path", help="Results log path.")
    parser.add_argument("--state-path", help="State JSON path.")
    parser.add_argument("--recent-window", type=int, default=8, help="How many recent rows to inspect.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.repo is not None:
        repo = resolve_repo_path(args.repo)
        results_path = resolve_repo_relative(repo, args.results_path, repo / "research-results.tsv")
        state_path = resolve_repo_relative(
            repo,
            args.state_path,
            resolve_state_path_for_log(args.state_path, None, cwd=repo),
        )
    else:
        results_path = Path(args.results_path or "research-results.tsv")
        state_path = Path(args.state_path or "autoresearch-state.json")

    print(
        json.dumps(
            evaluate_direction_shift(
                results_path=results_path,
                state_path=state_path,
                recent_window=args.recent_window,
            ),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AutoresearchError as exc:
        raise SystemExit(f"error: {exc}")
