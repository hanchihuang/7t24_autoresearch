#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Initialize a generic 7x24 autoresearch launch file from the template."
    )
    parser.add_argument("--project-dir", required=True, help="Absolute path to the target project.")
    parser.add_argument("--output", required=True, help="Where to write the generated launch json.")
    parser.add_argument("--goal", required=True, help="Measurable experiment goal.")
    parser.add_argument("--metric", required=True, help="Metric name.")
    parser.add_argument("--direction", choices=["higher", "lower"], required=True, help="Optimization direction.")
    parser.add_argument("--verify", required=True, help="Mechanical verification command.")
    parser.add_argument("--guard", required=True, help="Cheap guard command.")
    parser.add_argument("--run-tag", required=True, help="Run tag.")
    parser.add_argument("--scope", required=True, help="Comma-separated scope patterns.")
    parser.add_argument("--prompt", required=True, help="Original prompt / operator instruction.")
    parser.add_argument(
        "--session-mode",
        choices=["background", "foreground"],
        default="background",
        help="Session mode.",
    )
    parser.add_argument(
        "--web-search",
        choices=["enabled", "disabled"],
        default="disabled",
        help="Whether web search is enabled for the run.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    template_path = root / "templates" / "autoresearch-task.template.json"
    payload = json.loads(template_path.read_text(encoding="utf-8"))

    project_dir = str(Path(args.project_dir).resolve())
    output_path = Path(args.output).resolve()
    created_at = now_iso()

    payload["config"]["goal"] = args.goal
    payload["config"]["metric"] = args.metric
    payload["config"]["direction"] = args.direction
    payload["config"]["verify"] = args.verify
    payload["config"]["guard"] = args.guard
    payload["config"]["run_tag"] = args.run_tag
    payload["config"]["scope"] = args.scope
    payload["config"]["session_mode"] = args.session_mode
    payload["config"]["web_search"] = args.web_search
    payload["config"]["repos"] = [
        {
            "path": project_dir,
            "role": "primary",
            "scope": args.scope,
        }
    ]

    payload["created_at"] = created_at
    payload["updated_at"] = created_at
    payload["original_goal"] = args.prompt
    payload["prompt_text"] = args.prompt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(str(output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
