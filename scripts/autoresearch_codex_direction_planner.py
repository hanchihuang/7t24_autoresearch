#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from autoresearch_direction_advisor import evaluate_direction_shift
from autoresearch_helpers import (
    AutoresearchError,
    command_is_executable,
    default_lessons_path,
    resolve_repo_path,
    resolve_repo_relative,
    resolve_state_path_for_log,
    write_json_atomic,
    read_state_payload,
)


DEFAULT_RESULTS_PATH = "research-results.tsv"
DEFAULT_OUTPUT_MD = "autoresearch-next-directions.md"
DEFAULT_OUTPUT_JSON = "autoresearch-next-directions.json"


def build_prompt(
    *,
    repo: Path,
    results_path: Path,
    state_path: Path,
    lessons_path: Path,
    advisor_payload: dict[str, Any],
) -> str:
    suggested = ", ".join(advisor_payload.get("suggested_directions", [])) or "<none>"
    exhausted = ", ".join(advisor_payload.get("exhausted_families", [])) or "<none>"
    rationales = "\n".join(f"- {item}" for item in advisor_payload.get("rationales", [])) or "- <none>"
    return f"""
You are helping plan the next hypothesis families for an autoresearch loop.

Repo: {repo}
Results log: {results_path}
State file: {state_path}
Lessons file: {lessons_path}

Rule-based direction advisor summary:
- should_shift_direction: {advisor_payload.get("should_shift_direction")}
- shift_reason: {advisor_payload.get("shift_reason")}
- exhausted_families: {exhausted}
- suggested_directions: {suggested}
- rationales:
{rationales}

Task:
1. Read the results log, state file, and lessons file.
2. Infer the best next hypothesis families for the run.
3. Prefer genuinely new gain sources over local parameter churn.
4. Be specific and pragmatic.
5. Return STRICT JSON only. No markdown fencing, no commentary.

Required JSON schema:
{{
  "summary": "short paragraph",
  "why_now": ["reason 1", "reason 2"],
  "deprioritize": ["family a", "family b"],
  "next_directions": [
    {{
      "name": "direction name",
      "priority": 1,
      "rationale": "why this is worth trying now",
      "candidate_signals": ["signal 1", "signal 2"],
      "scout_idea": "small scout experiment idea"
    }}
  ]
}}
""".strip() + "\n"


def render_markdown(plan: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Next Directions")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(str(plan.get("summary", "")).strip() or "<empty>")
    why_now = plan.get("why_now", [])
    if why_now:
      lines.append("")
      lines.append("## Why Now")
      lines.append("")
      for item in why_now:
          lines.append(f"- {item}")
    deprioritize = plan.get("deprioritize", [])
    if deprioritize:
      lines.append("")
      lines.append("## Deprioritize")
      lines.append("")
      for item in deprioritize:
          lines.append(f"- {item}")
    directions = plan.get("next_directions", [])
    if directions:
      lines.append("")
      lines.append("## Next Directions")
      lines.append("")
      for item in directions:
          lines.append(f"### {item.get('priority', '?')}. {item.get('name', '<unnamed>')}")
          lines.append("")
          lines.append(str(item.get("rationale", "")).strip() or "<no rationale>")
          signals = item.get("candidate_signals", [])
          if signals:
              lines.append("")
              lines.append("Candidate signals:")
              for signal in signals:
                  lines.append(f"- {signal}")
          scout = str(item.get("scout_idea", "")).strip()
          if scout:
              lines.append("")
              lines.append(f"Scout idea: {scout}")
          lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def apply_direction_preferences(
    plan: dict[str, Any],
    *,
    preferred: list[str],
    banned: list[str],
) -> dict[str, Any]:
    directions = plan.get("next_directions", [])
    if not isinstance(directions, list):
        return plan
    preferred_set = {item.strip() for item in preferred if item.strip()}
    banned_set = {item.strip() for item in banned if item.strip()}
    filtered: list[dict[str, Any]] = []
    for item in directions:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if name in banned_set:
            continue
        filtered.append(item)
    if preferred_set:
        filtered.sort(
            key=lambda item: (
                0 if str(item.get("name", "")).strip() in preferred_set else 1,
                int(item.get("priority", 999) or 999),
            )
        )
    for index, item in enumerate(filtered, start=1):
        item["priority"] = index
    plan["next_directions"] = filtered
    return plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use Codex to plan the next hypothesis families after an autoresearch run soft-blocks."
    )
    parser.add_argument("--repo", required=True, help="Primary repo root.")
    parser.add_argument("--results-path", help="Results log path.")
    parser.add_argument("--state-path", help="State file path.")
    parser.add_argument("--lessons-path", help="Lessons file path.")
    parser.add_argument("--output-md", help="Markdown output path.")
    parser.add_argument("--output-json", help="JSON output path.")
    parser.add_argument("--codex-bin", default="codex", help="Codex executable.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = resolve_repo_path(args.repo)
    results_path = resolve_repo_relative(repo, args.results_path, repo / DEFAULT_RESULTS_PATH)
    state_path = resolve_repo_relative(
        repo,
        args.state_path,
        resolve_state_path_for_log(args.state_path, None, cwd=repo),
    )
    lessons_path = resolve_repo_relative(repo, args.lessons_path, default_lessons_path(repo))
    output_md = resolve_repo_relative(repo, args.output_md, repo / DEFAULT_OUTPUT_MD)
    output_json = resolve_repo_relative(repo, args.output_json, repo / DEFAULT_OUTPUT_JSON)

    if not command_is_executable(args.codex_bin):
        raise AutoresearchError(f"Codex executable is not available: {args.codex_bin}")

    state_payload = read_state_payload(state_path)
    config = state_payload.get("config", {})
    preferred = list(config.get("preferred_direction_families", []) or [])
    banned = list(config.get("banned_direction_families", []) or [])

    advisor_payload = evaluate_direction_shift(
        results_path=results_path,
        state_path=state_path,
        recent_window=8,
    )
    prompt = build_prompt(
        repo=repo,
        results_path=results_path,
        state_path=state_path,
        lessons_path=lessons_path,
        advisor_payload=advisor_payload,
    )
    cmd = [args.codex_bin, "exec", "-C", str(repo), "-"]
    completed = subprocess.run(
        cmd,
        cwd=repo,
        input=prompt,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise AutoresearchError(
            f"Codex direction planning failed: {completed.stderr.strip() or completed.stdout.strip()}"
        )

    raw = completed.stdout.strip()
    try:
        plan = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AutoresearchError(f"Codex planner returned invalid JSON: {exc}: {raw[:500]}") from exc

    plan = apply_direction_preferences(
        plan,
        preferred=preferred,
        banned=banned,
    )
    plan["preferred_direction_families"] = preferred
    plan["banned_direction_families"] = banned

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(render_markdown(plan), encoding="utf-8")
    write_json_atomic(output_json, plan)

    print(
        json.dumps(
            {
                "output_md": str(output_md),
                "output_json": str(output_json),
                "advisor_payload": advisor_payload,
            },
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
