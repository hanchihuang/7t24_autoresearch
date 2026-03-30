# GRPO GSM8K Snapshot

This folder is the exported workspace snapshot for the unattended GRPO-on-GSM8K autoresearch loop.

## What is here

- `llama3_1_(8b)_grpo.py`: main GRPO / eval script used by the run.
- `grpo_autoresearch_watchdog.py`: local watchdog that monitors the runtime and restarts it when appropriate.
- `autoresearch-launch.json`: launch manifest for the background loop.
- `autoresearch-state.json`: latest state snapshot.
- `autoresearch-runtime.json`: latest runtime snapshot.
- `autoresearch-lessons.md`: accumulated lessons and pivots.
- `research-results.tsv`: iteration-by-iteration experiment log.
- `grpo_autoresearch_progress_iter75_88.md`: Chinese progress summary focused on iter75 through iter88.

## Included run summaries

- `gsm8k_improved/autoresearch_confirm200_masktrunc_iter75c/`: current best retained `0.48 (96/200)`.
- `gsm8k_improved/autoresearch_scout_iter85_percentage_focus/`: failed narrowing experiment, `0.20`.
- `gsm8k_improved/autoresearch_scout_iter87_difficulty_prompt_replay/`: failed replay approximation, `0.20`.

## Current status

- Best retained metric: `0.48`
- Best iteration: `75`
- Current recorded iteration: `88`
- Current state: `pivot`
- Supervisor recommendation: `needs_human`

## Upstream engine base

The runtime engine in this repository was copied from the local `codex-autoresearch` skill checkout at commit `0bb284a`.
