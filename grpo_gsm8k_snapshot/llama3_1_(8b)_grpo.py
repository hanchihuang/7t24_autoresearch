# -*- coding: utf-8 -*-
"""
Local-runnable GRPO example adapted from the original Unsloth notebook.

Why this file differs from the Colab notebook:
1. It does not require `unsloth` or `vllm` by default.
2. It supports both a lightweight smoke test and a more realistic local
   training configuration.
3. It keeps the same overall flow: prepare data -> define rewards -> run GRPO
   -> evaluate -> save the result.

Environment variables you can override:
    SMOKE_TEST=1/0
    MODEL_NAME=Qwen/Qwen2.5-0.5B-Instruct
    MAX_STEPS=1
    OUTPUT_DIR=outputs_llama3_1_grpo
"""

from __future__ import annotations

import os
import re
import json
import pickle
import random
import gc
import inspect
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, DownloadConfig, load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from trl import GRPOConfig, GRPOTrainer


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# 读取环境变量中的布尔开关，支持 1/true/yes/y 这几种常见写法。
def env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "y"}


def env_present(name: str) -> bool:
    return os.getenv(name) is not None


def env_int(name: str) -> int | None:
    value = os.getenv(name)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def infer_scout_like_run() -> bool:
    dataset_source = os.getenv("DATASET_SOURCE", "").strip().lower()
    max_steps = env_int("MAX_STEPS")
    num_eval_samples = env_int("NUM_EVAL_SAMPLES")
    return (
        dataset_source == "gsm8k"
        and max_steps is not None
        and max_steps <= 20
        and num_eval_samples is not None
        and num_eval_samples <= 30
    )


def infer_smoke_test() -> bool:
    if env_present("SMOKE_TEST"):
        return env_flag("SMOKE_TEST", "1")
    single_run_signals = [
        "DATASET_SOURCE",
        "MAX_STEPS",
        "MAX_TRAIN_SAMPLES",
        "NUM_EVAL_SAMPLES",
        "SFT_WARMUP_STEPS",
        "EVAL_ONLY",
        "ADAPTER_PATH",
        "EVAL_USE_CONFIDENCE_RERANK",
        "EVAL_NUM_CANDIDATES",
    ]
    # If the caller already supplied real-run knobs, prefer the non-smoke path.
    if any(env_present(name) for name in single_run_signals):
        return False
    return True


def infer_run_mode() -> str:
    explicit = os.getenv("RUN_MODE")
    if explicit is not None:
        return explicit.strip().lower()
    single_run_signals = [
        "TRAINING_METHOD",
        "DATASET_SOURCE",
        "MAX_STEPS",
        "MAX_TRAIN_SAMPLES",
        "NUM_EVAL_SAMPLES",
        "SFT_WARMUP_STEPS",
        "EVAL_ONLY",
        "ADAPTER_PATH",
        "EVAL_USE_CONFIDENCE_RERANK",
        "EVAL_NUM_CANDIDATES",
    ]
    # Default to one explicit experiment when the environment already looks like a
    # configured scout/confirm invocation instead of a local exploratory launch.
    if any(env_present(name) for name in single_run_signals):
        return "single"
    return "auto"


def infer_run_protocol() -> str:
    explicit = os.getenv("RUN_PROTOCOL")
    if explicit is not None:
        normalized = explicit.strip().lower()
        if normalized in {"scout", "confirm", "standard"}:
            return normalized
    return "scout" if infer_scout_like_run() else "standard"


# 默认走 smoke test，这样本地第一次执行能快速验证流程是否完整。
SMOKE_TEST = infer_smoke_test()
SCOUT_LIKE_RUN = infer_scout_like_run()
RUN_PROTOCOL = infer_run_protocol()
IS_SCOUT_PROTOCOL = RUN_PROTOCOL == "scout"
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "outputs_llama3_1_grpo"))
MAX_STEPS = int(os.getenv("MAX_STEPS", "1" if SMOKE_TEST else "100"))
MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", "256" if SMOKE_TEST else "512"))
MAX_PROMPT_LENGTH = int(os.getenv("MAX_PROMPT_LENGTH", "128" if SMOKE_TEST else "256"))
MAX_COMPLETION_LENGTH = MAX_SEQ_LENGTH - MAX_PROMPT_LENGTH
EVAL_MAX_NEW_TOKENS = int(
    os.getenv(
        "EVAL_MAX_NEW_TOKENS",
        "96" if SCOUT_LIKE_RUN else ("128" if SMOKE_TEST else "256"),
    )
)
NUM_GENERATIONS = int(os.getenv("NUM_GENERATIONS", "2" if SMOKE_TEST else "4"))
PER_DEVICE_TRAIN_BATCH_SIZE = int(
    os.getenv("PER_DEVICE_TRAIN_BATCH_SIZE", str(NUM_GENERATIONS))
)
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "5e-6" if SMOKE_TEST else "2e-5"))
GRPO_TEMPERATURE = float(os.getenv("GRPO_TEMPERATURE", "1.0"))
GRPO_TOP_P = float(os.getenv("GRPO_TOP_P", "1.0"))
grpo_top_k_env = os.getenv("GRPO_TOP_K", "").strip().lower()
GRPO_TOP_K = None if grpo_top_k_env in {"", "none"} else int(grpo_top_k_env)
GRPO_SCALE_REWARDS = os.getenv("GRPO_SCALE_REWARDS", "group").strip().lower() or "group"
GRPO_LOSS_TYPE = os.getenv("GRPO_LOSS_TYPE", "dapo").strip().lower() or "dapo"
SAVE_STEPS = int(os.getenv("SAVE_STEPS", str(MAX_STEPS if SMOKE_TEST else max(20, MAX_STEPS // 5))))
DATASET_SOURCE = os.getenv("DATASET_SOURCE", "local" if SMOKE_TEST else "gsm8k")
DATASET_SPLIT = os.getenv("DATASET_SPLIT", "train")
DATASET_CONFIG = os.getenv("DATASET_CONFIG", "main")
DATASET_START_INDEX = int(os.getenv("DATASET_START_INDEX", "0"))
MAX_TRAIN_SAMPLES = int(os.getenv("MAX_TRAIN_SAMPLES", "3" if SMOKE_TEST else "1000"))
NUM_EVAL_SAMPLES = int(os.getenv("NUM_EVAL_SAMPLES", "5" if SMOKE_TEST else "20"))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
EVAL_EVERY_STEPS = int(os.getenv("EVAL_EVERY_STEPS", "5" if SMOKE_TEST else "20"))
PROGRESS_LOG_EVERY = int(os.getenv("PROGRESS_LOG_EVERY", "1" if SMOKE_TEST else "10"))
SYNTHETIC_DIFFICULTY = os.getenv("SYNTHETIC_DIFFICULTY", "medium" if SMOKE_TEST else "hard")
SYNTHETIC_FOCUS = os.getenv("SYNTHETIC_FOCUS", "").strip().lower()
SYNTHETIC_PROFILE = os.getenv("SYNTHETIC_PROFILE", "targeted_gsm8k" if DATASET_SOURCE == "gsm8k" else "uniform").strip().lower()
MINE_CHALLENGING_SYNTHETIC = env_flag(
    "MINE_CHALLENGING_SYNTHETIC",
    "0" if SMOKE_TEST or SCOUT_LIKE_RUN else "1",
)
CANDIDATE_POOL_MULTIPLIER = int(os.getenv("CANDIDATE_POOL_MULTIPLIER", "4"))
SYNTHETIC_AUGMENT_COUNT = env_int("SYNTHETIC_AUGMENT_COUNT")
ANCHOR_REPLAY_COUNT = env_int("ANCHOR_REPLAY_COUNT")
SYNTHETIC_SFT_ONLY = env_flag("SYNTHETIC_SFT_ONLY", "0")
ENABLE_EQUIV_SYNTHETIC_TRANSFORMS = env_flag("ENABLE_EQUIV_SYNTHETIC_TRANSFORMS", "0")
TEACHER_COMPLETION_BANK_PATH = os.getenv("TEACHER_COMPLETION_BANK_PATH", "").strip()
ENABLE_TEACHER_COMPLETION_REPLAY = env_flag("ENABLE_TEACHER_COMPLETION_REPLAY", "0")
TEACHER_REPLAY_COUNT = env_int("TEACHER_REPLAY_COUNT")
ENABLE_GRPO_TEACHER_ANCHOR = env_flag("ENABLE_GRPO_TEACHER_ANCHOR", "0")
REWARD_WEIGHT_TEACHER_ANCHOR = float(os.getenv("REWARD_WEIGHT_TEACHER_ANCHOR", "0.0"))
TEACHER_REPLAY_SLICES = tuple(
    slice_name.strip().lower()
    for slice_name in os.getenv(
        "TEACHER_REPLAY_SLICES",
        "percentage,rate_or_ratio",
    ).split(",")
    if slice_name.strip()
)
ENABLE_DYNAMIC_TEACHER_REPLAY = env_flag("ENABLE_DYNAMIC_TEACHER_REPLAY", "0")
DYNAMIC_TEACHER_REPLAY_COUNT = env_int("DYNAMIC_TEACHER_REPLAY_COUNT")
DYNAMIC_TEACHER_REPLAY_SLICES = tuple(
    slice_name.strip().lower()
    for slice_name in os.getenv(
        "DYNAMIC_TEACHER_REPLAY_SLICES",
        "percentage,rate_or_ratio",
    ).split(",")
    if slice_name.strip()
)
ENABLE_PROMPT_REPLAY = env_flag("ENABLE_PROMPT_REPLAY", "0")
PROMPT_REPLAY_COUNT = env_int("PROMPT_REPLAY_COUNT")
PROMPT_REPLAY_SLICES = tuple(
    slice_name.strip().lower()
    for slice_name in os.getenv(
        "PROMPT_REPLAY_SLICES",
        "percentage,rate_or_ratio",
    ).split(",")
    if slice_name.strip()
)
BUILD_TEACHER_BANK_ONLY = env_flag("BUILD_TEACHER_BANK_ONLY", "0")
TEACHER_BANK_TARGET_COUNT = env_int("TEACHER_BANK_TARGET_COUNT")
TEACHER_BANK_CANDIDATE_CAP = env_int("TEACHER_BANK_CANDIDATE_CAP")
TEACHER_BANK_OUTPUT_PATH = os.getenv("TEACHER_BANK_OUTPUT_PATH", "").strip()
TEACHER_BANK_SLICES = tuple(
    slice_name.strip().lower()
    for slice_name in os.getenv(
        "TEACHER_BANK_SLICES",
        ",".join(DYNAMIC_TEACHER_REPLAY_SLICES) or "percentage,rate_or_ratio",
    ).split(",")
    if slice_name.strip()
)

default_sft_warmup_steps = "0"
if DATASET_SOURCE == "synthetic" and SYNTHETIC_DIFFICULTY == "hard":
    default_sft_warmup_steps = "8" if SMOKE_TEST else "20"
if DATASET_SOURCE == "synthetic" and SYNTHETIC_FOCUS == "mixed_expression":
    default_sft_warmup_steps = "30" if SMOKE_TEST else "60"
SFT_WARMUP_STEPS = int(os.getenv("SFT_WARMUP_STEPS", default_sft_warmup_steps))
TRAINING_METHOD = os.getenv("TRAINING_METHOD", "auto").strip().lower()
RUN_MODE = infer_run_mode()
AUTO_IMPROVE_IF_NO_GAIN = env_flag("AUTO_IMPROVE_IF_NO_GAIN", "0" if SMOKE_TEST else "1")
AUTO_MAX_ATTEMPTS = int(os.getenv("AUTO_MAX_ATTEMPTS", "3"))
MIN_GRPO_GAIN = float(os.getenv("MIN_GRPO_GAIN", "0.01"))
SFT_BASELINE_STEPS = int(os.getenv("SFT_BASELINE_STEPS", str(max(MAX_STEPS, SFT_WARMUP_STEPS))))
FAIR_COMPARE_TOTAL_STEPS = env_flag("FAIR_COMPARE_TOTAL_STEPS", "1")
TARGETED_GSM8K_SCOUT = (
    DATASET_SOURCE == "gsm8k"
    and IS_SCOUT_PROTOCOL
    and SYNTHETIC_PROFILE == "targeted_gsm8k"
)
REWARD_WEIGHT_XML = float(os.getenv("REWARD_WEIGHT_XML", "1.0"))
REWARD_WEIGHT_NUMERIC = float(os.getenv("REWARD_WEIGHT_NUMERIC", "1.0"))
REWARD_WEIGHT_DISTANCE = float(os.getenv("REWARD_WEIGHT_DISTANCE", "1.0"))
REWARD_WEIGHT_PARTIAL = float(os.getenv("REWARD_WEIGHT_PARTIAL", "1.0"))
REWARD_WEIGHT_REASONING = float(os.getenv("REWARD_WEIGHT_REASONING", "1.0"))
REWARD_WEIGHT_EQUATION = float(os.getenv("REWARD_WEIGHT_EQUATION", "0.0"))
REWARD_WEIGHT_BREVITY = float(
    os.getenv("REWARD_WEIGHT_BREVITY", "0.08" if TARGETED_GSM8K_SCOUT else "0.0")
)
REWARD_WEIGHT_STEP_ALIGN = float(os.getenv("REWARD_WEIGHT_STEP_ALIGN", "0.0"))
REWARD_WEIGHT_WRONG_PENALTY = float(
    os.getenv("REWARD_WEIGHT_WRONG_PENALTY", "0.12" if TARGETED_GSM8K_SCOUT else "0.0")
)
REWARD_WEIGHT_CORRECTNESS = float(os.getenv("REWARD_WEIGHT_CORRECTNESS", "1.0"))
REWARD_WEIGHT_VERIFIER = float(os.getenv("REWARD_WEIGHT_VERIFIER", "0.0"))
SKIP_EVAL_BEFORE = env_flag("SKIP_EVAL_BEFORE", "1" if SCOUT_LIKE_RUN else "0")
SKIP_SAMPLE_GENERATION = env_flag("SKIP_SAMPLE_GENERATION", "1" if SCOUT_LIKE_RUN else "0")
DISABLE_HELDOUT_CALLBACK = env_flag("DISABLE_HELDOUT_CALLBACK", "1" if SCOUT_LIKE_RUN else "0")
SKIP_EVAL_WARMUP = env_flag("SKIP_EVAL_WARMUP", "1" if IS_SCOUT_PROTOCOL else "0")
SAVE_ADAPTER = env_flag("SAVE_ADAPTER", "0" if IS_SCOUT_PROTOCOL else "1")
WRITE_EXPLANATION = env_flag("WRITE_EXPLANATION", "0" if IS_SCOUT_PROTOCOL else "1")
ADAPTER_PATH = os.getenv("ADAPTER_PATH", "").strip()
EVAL_USE_CONFIDENCE_RERANK = env_flag("EVAL_USE_CONFIDENCE_RERANK", "0")
EVAL_NUM_CANDIDATES = int(os.getenv("EVAL_NUM_CANDIDATES", "4"))
EVAL_RERANK_TEMPERATURE = float(os.getenv("EVAL_RERANK_TEMPERATURE", "0.7"))
EVAL_RERANK_TOP_P = float(os.getenv("EVAL_RERANK_TOP_P", "0.95"))
CONFIDENCE_WEIGHT = float(os.getenv("CONFIDENCE_WEIGHT", "1.0"))
CONSENSUS_WEIGHT = float(os.getenv("CONSENSUS_WEIGHT", "0.35"))
FORMAT_WEIGHT = float(os.getenv("FORMAT_WEIGHT", "0.15"))
LOW_CONFIDENCE_PROB_THRESHOLD = float(os.getenv("LOW_CONFIDENCE_PROB_THRESHOLD", "0.25"))
LOW_CONFIDENCE_WEIGHT = float(os.getenv("LOW_CONFIDENCE_WEIGHT", "0.35"))
NOVELTY_WEIGHT = float(os.getenv("NOVELTY_WEIGHT", "0.1"))
ANSWER_AGG_COUNT_WEIGHT = float(os.getenv("ANSWER_AGG_COUNT_WEIGHT", "0.85"))
ANSWER_AGG_STRICT_WEIGHT = float(os.getenv("ANSWER_AGG_STRICT_WEIGHT", "0.2"))
ANSWER_AGG_EQUATION_WEIGHT = float(os.getenv("ANSWER_AGG_EQUATION_WEIGHT", "0.25"))
ANSWER_AGG_DIVERSITY_WEIGHT = float(os.getenv("ANSWER_AGG_DIVERSITY_WEIGHT", "0.08"))
ANSWER_AGG_LOW_CONF_WEIGHT = float(os.getenv("ANSWER_AGG_LOW_CONF_WEIGHT", "0.2"))
ANSWER_AGG_MIN_GROUP_SIZE = int(os.getenv("ANSWER_AGG_MIN_GROUP_SIZE", "2"))
ANSWER_AGG_MARGIN = float(os.getenv("ANSWER_AGG_MARGIN", "0.24"))
ANSWER_AGG_PAIR_COUNT_WEIGHT = float(os.getenv("ANSWER_AGG_PAIR_COUNT_WEIGHT", "0.45"))
ANSWER_AGG_PAIR_MAX_SINGLE_GAP = float(os.getenv("ANSWER_AGG_PAIR_MAX_SINGLE_GAP", "0.12"))
REWARD_WEIGHT_NOVELTY = float(os.getenv("REWARD_WEIGHT_NOVELTY", "0.05"))
MINE_NUM_CANDIDATES = int(os.getenv("MINE_NUM_CANDIDATES", "4"))
MINE_TEMPERATURE = float(os.getenv("MINE_TEMPERATURE", "0.8"))
MINE_TOP_P = float(os.getenv("MINE_TOP_P", "0.95"))
EVAL_ONLY = env_flag("EVAL_ONLY", "0")
default_anchor_replay_enabled = (
    "1"
    if ADAPTER_PATH and TARGETED_GSM8K_SCOUT and SYNTHETIC_DIFFICULTY == "hard" and not EVAL_ONLY
    else "0"
)
ENABLE_ANCHOR_REPLAY = env_flag("ENABLE_ANCHOR_REPLAY", default_anchor_replay_enabled)
ANCHOR_REPLAY_SLICES = tuple(
    slice_name.strip().lower()
    for slice_name in os.getenv(
        "ANCHOR_REPLAY_SLICES",
        "basic_arithmetic,difference,multi_number",
    ).split(",")
    if slice_name.strip()
)
CONTINUATION_SAFE_DYNAMICS = env_flag(
    "CONTINUATION_SAFE_DYNAMICS",
    "1" if ADAPTER_PATH and TARGETED_GSM8K_SCOUT and not EVAL_ONLY else "0",
)
ENABLE_SFT_TEACHER_BANK_PRIORITY = env_flag("ENABLE_SFT_TEACHER_BANK_PRIORITY", "0")
SFT_TEACHER_BANK_TARGET = env_int("SFT_TEACHER_BANK_TARGET")
SFT_TEACHER_BANK_KEEP_AUX_COUNT = env_int("SFT_TEACHER_BANK_KEEP_AUX_COUNT")
SFT_TEACHER_BANK_SLICES = tuple(
    slice_name.strip().lower()
    for slice_name in os.getenv(
        "SFT_TEACHER_BANK_SLICES",
        "percentage,rate_or_ratio",
    ).split(",")
    if slice_name.strip()
)
EXPERIMENT_HYPOTHESIS = os.getenv("EXPERIMENT_HYPOTHESIS", "").strip()
EXPERIMENT_RISK = os.getenv("EXPERIMENT_RISK", "").strip()
EXPERIMENT_NOTE = os.getenv("EXPERIMENT_NOTE", "").strip()
VERIFIER_BUNDLE_PATH = os.getenv("VERIFIER_BUNDLE_PATH", "").strip()
VERIFIER_SCORE_WEIGHT = float(os.getenv("VERIFIER_SCORE_WEIGHT", "0.2"))
VERIFIER_TIE_MARGIN = float(os.getenv("VERIFIER_TIE_MARGIN", "0.15"))
VERIFIER_MIN_CANDIDATES = int(os.getenv("VERIFIER_MIN_CANDIDATES", "2"))
VERIFIER_REQUIRE_ANSWER_DISAGREEMENT = env_flag("VERIFIER_REQUIRE_ANSWER_DISAGREEMENT", "1")
HF_LOCAL_FILES_ONLY = env_flag("HF_LOCAL_FILES_ONLY", "1")

if HF_LOCAL_FILES_ONLY:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

if CONTINUATION_SAFE_DYNAMICS and not env_present("LEARNING_RATE"):
    LEARNING_RATE = min(LEARNING_RATE, 2e-6)

GRPO_BETA = float(os.getenv("GRPO_BETA", "0.04" if CONTINUATION_SAFE_DYNAMICS else "0.0"))
GRPO_EPSILON = float(os.getenv("GRPO_EPSILON", "0.12" if CONTINUATION_SAFE_DYNAMICS else "0.2"))
GRPO_WARMUP_RATIO = float(os.getenv("GRPO_WARMUP_RATIO", "0.2" if CONTINUATION_SAFE_DYNAMICS else "0.1"))
GRPO_MAX_GRAD_NORM = float(
    os.getenv("GRPO_MAX_GRAD_NORM", "0.05" if CONTINUATION_SAFE_DYNAMICS else "0.1")
)
MASK_TRUNCATED_COMPLETIONS = env_flag("MASK_TRUNCATED_COMPLETIONS", "0")
TOP_ENTROPY_QUANTILE = float(os.getenv("TOP_ENTROPY_QUANTILE", "1.0"))
off_policy_mask_threshold_env = os.getenv("OFF_POLICY_MASK_THRESHOLD", "").strip()
OFF_POLICY_MASK_THRESHOLD = (
    float(off_policy_mask_threshold_env) if off_policy_mask_threshold_env else None
)
USE_BIAS_CORRECTION_KL = env_flag("USE_BIAS_CORRECTION_KL", "0")

_VERIFIER_BUNDLE_CACHE: dict[str, Any] | None = None
_VERIFIER_BUNDLE_CACHE_PATH = ""
_VERIFIER_BUNDLE_ERROR = ""

SYSTEM_PROMPT = """You are solving a short arithmetic word problem.
Always answer with exactly this XML structure:
<reasoning>
one short arithmetic explanation
</reasoning>
<answer>
final integer only
</answer>

Do not copy placeholder words like "one short arithmetic explanation" or "final integer only".
"""

LOCAL_SAMPLES = [
    {
        "question": "If John has 2 apples and buys 3 more, how many apples does he have?",
        "answer": "#### 5",
    },
    {
        "question": "What is 12 minus 7?",
        "answer": "#### 5",
    },
    {
        "question": "A box has 4 red balls and 6 blue balls. How many balls are there in total?",
        "answer": "#### 10",
    },
    {
        "question": "If a car travels 30 miles in the morning and 12 miles in the evening, how many miles did it travel in total?",
        "answer": "#### 42",
    },
    {
        "question": "There are 9 birds on a tree. 4 fly away. How many birds remain?",
        "answer": "#### 5",
    },
    {
        "question": "What is 7 multiplied by 6?",
        "answer": "#### 42",
    },
    {
        "question": "A class has 15 students and 3 more join. How many students are there now?",
        "answer": "#### 18",
    },
    {
        "question": "What is 20 divided by 4?",
        "answer": "#### 5",
    },
    {
        "question": "Mia has 11 stickers and gets 9 more. How many stickers does she have?",
        "answer": "#### 20",
    },
    {
        "question": "What is 14 plus 8?",
        "answer": "#### 22",
    },
    {
        "question": "A farmer has 25 sheep and sells 7. How many sheep remain?",
        "answer": "#### 18",
    },
    {
        "question": "What is 9 plus 9?",
        "answer": "#### 18",
    },
]

LOCAL_EVAL_SAMPLES = [
    {"question": "Calculate 2 + 3.", "answer": "5"},
    {"question": "What is 16 minus 9?", "answer": "7"},
    {"question": "What is 8 times 5?", "answer": "40"},
    {"question": "A jar has 13 marbles and 6 are added. How many marbles are in the jar now?", "answer": "19"},
    {"question": "What is 21 divided by 3?", "answer": "7"},
    {"question": "What is 17 plus 26?", "answer": "43"},
    {"question": "What is 45 minus 18?", "answer": "27"},
    {"question": "What is 9 times 7?", "answer": "63"},
    {"question": "A shop sold 18 apples in the morning and 15 in the afternoon. How many apples were sold in total?", "answer": "33"},
    {"question": "There were 30 cookies and 11 were eaten. How many are left?", "answer": "19"},
    {"question": "What is 54 divided by 6?", "answer": "9"},
    {"question": "Sam has 7 packs of cards with 8 cards each. How many cards does he have?", "answer": "56"},
    {"question": "What is 19 plus 14?", "answer": "33"},
    {"question": "What is 100 minus 37?", "answer": "63"},
    {"question": "A bus has 22 people and 13 more get on. How many people are on the bus?", "answer": "35"},
    {"question": "What is 6 times 12?", "answer": "72"},
    {"question": "What is 81 divided by 9?", "answer": "9"},
    {"question": "Lily had 28 stickers and gave away 9. How many stickers does she have now?", "answer": "19"},
    {"question": "What is 13 plus 29?", "answer": "42"},
    {"question": "What is 72 minus 25?", "answer": "47"},
    {"question": "What is 11 times 11?", "answer": "121"},
    {"question": "A library has 40 red books and 35 blue books. How many books are there altogether?", "answer": "75"},
    {"question": "What is 96 divided by 12?", "answer": "8"},
    {"question": "A farmer plants 14 rows with 5 trees in each row. How many trees are planted?", "answer": "70"},
    {"question": "What is 33 plus 18?", "answer": "51"},
]


# 统一的数值片段匹配规则：
# - 支持负数
# - 支持千分位逗号，如 1,234
# - 支持小数，如 12.0
# - 支持简单分数，如 6/3
NUMBER_PATTERN = re.compile(
    r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:/\d+)?"
)


# 从 <answer>...</answer> 中抽取最终答案；如果没有 XML，就回退到全文中的最后一个数值片段。
def extract_xml_answer(text: str) -> str:
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()

    numbers = NUMBER_PATTERN.findall(text)
    if numbers:
        return numbers[-1]
    return text.strip()


def canonicalize_numeric_text(text: str) -> str:
    candidate = text.strip()
    if not candidate:
        return ""

    candidate = candidate.replace(",", "")

    if "/" in candidate:
        try:
            value = Fraction(candidate)
            return str(value.numerator) if value.denominator == 1 else str(value)
        except (ValueError, ZeroDivisionError):
            pass

    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", candidate):
        try:
            value = float(candidate)
            return str(int(value)) if value.is_integer() else format(value, "g")
        except ValueError:
            pass

    return candidate


def normalize_answer(text: str) -> str:
    matches = NUMBER_PATTERN.findall(text)
    if matches:
        return canonicalize_numeric_text(matches[-1])
    return text.strip()


def has_answer_tag(text: str) -> bool:
    return bool(re.search(r"<answer>\s*.*?\s*</answer>", text, flags=re.DOTALL))


def has_reasoning_open_tag(text: str) -> bool:
    return "<reasoning>" in text


def has_strict_xml_structure(text: str) -> bool:
    reasoning_match = re.search(
        r"<reasoning>\s*(.*?)\s*</reasoning>",
        text,
        flags=re.DOTALL,
    )
    answer_match = re.search(
        r"<answer>\s*(.*?)\s*</answer>",
        text,
        flags=re.DOTALL,
    )
    if not reasoning_match or not answer_match:
        return False

    return (
        reasoning_match.start() <= answer_match.start()
        and reasoning_match.end() <= answer_match.start()
    )


def xml_format_score(text: str) -> float:
    if has_strict_xml_structure(text):
        return 0.2
    if has_reasoning_open_tag(text) and has_answer_tag(text):
        return 0.05
    if has_answer_tag(text):
        return 0.02
    return 0.0


def build_user_prompt(question: str, hint: str = "") -> str:
    hint_block = f"\nHint: {hint.strip()}" if hint.strip() else ""
    return (
        f"{question}{hint_block}\n"
        "Answer using <reasoning> and <answer> tags. "
        "Put only the final integer inside <answer>."
    )


def build_gold_reasoning(question: str, answer: str) -> str:
    normalized_answer = normalize_answer(answer)
    numeric_answer = normalized_answer if re.fullmatch(r"-?\d+", normalized_answer) else None
    text = question.strip()

    patterns: list[tuple[str, Any]] = [
        (
            r"^What is (\d+) plus (\d+)\?$",
            lambda m: f"{m.group(1)} + {m.group(2)} = {normalized_answer}.",
        ),
        (
            r"^What is (\d+) minus (\d+)\?$",
            lambda m: f"{m.group(1)} - {m.group(2)} = {normalized_answer}.",
        ),
        (
            r"^What is (\d+) times (\d+)\?$",
            lambda m: f"{m.group(1)} * {m.group(2)} = {normalized_answer}.",
        ),
        (
            r"^What is (\d+) multiplied by (\d+)\?$",
            lambda m: f"{m.group(1)} * {m.group(2)} = {normalized_answer}.",
        ),
        (
            r"^What is (\d+) divided by (\d+)\?$",
            lambda m: f"{m.group(1)} / {m.group(2)} = {normalized_answer}.",
        ),
        (
            r"^Calculate (\d+) \+ (\d+)\.$",
            lambda m: f"{m.group(1)} + {m.group(2)} = {normalized_answer}.",
        ),
        (
            r"^Compute \((\d+) \+ (\d+)\) \* (\d+) - (\d+)\.$",
            lambda m: (
                f"First {m.group(1)} + {m.group(2)} = {int(m.group(1)) + int(m.group(2))}, "
                f"then multiply by {m.group(3)} and subtract {m.group(4)} to get {normalized_answer}."
            ),
        ),
        (
            r"^Compute (\d+) divided by (\d+), then add (\d+)\.$",
            lambda m: (
                f"First {m.group(1)} / {m.group(2)} = {int(m.group(1)) // int(m.group(2))}, "
                f"then add {m.group(3)} to get {normalized_answer}."
            ),
        ),
        (
            r"^There are (\d+) \w+ with (\d+) items in each\. After packing, (\d+) extra items are added\. How many items are there in total\?$",
            lambda m: (
                f"{m.group(1)} * {m.group(2)} = {int(m.group(1)) * int(m.group(2))}, "
                f"then add {m.group(3)} to get {normalized_answer}."
            ),
        ),
        (
            r"^\w+ collected (\d+) packs of \w+ with (\d+) in each pack\. Later \w+ gave away (\d+) and then found (\d+) more\. How many \w+ does \w+ have now\?$",
            lambda m: (
                f"{m.group(1)} * {m.group(2)} = {int(m.group(1)) * int(m.group(2))}, "
                f"then subtract {m.group(3)} and add {m.group(4)} to get {normalized_answer}."
            ),
        ),
        (
            r"^\w+ had (\d+) \w+\. Then \w+ bought (\d+) more and later gave away (\d+)\. How many \w+ does \w+ have now\?$",
            lambda m: (
                f"Start with {m.group(1)}, add {m.group(2)}, then subtract {m.group(3)} to get {normalized_answer}."
            ),
        ),
        (
            r"^\w+ organizes (\d+) groups of \w+ with (\d+) in each group\. Then \w+ adds (\d+) more \w+\. How many \w+ are there in total\?$",
            lambda m: (
                f"{m.group(1)} * {m.group(2)} = {int(m.group(1)) * int(m.group(2))}, "
                f"then add {m.group(3)} to get {normalized_answer}."
            ),
        ),
        (
            r"^A club had (\d+) points\. Then the score increased by (\d+)%\. What is the new total number of points\?$",
            lambda m: (
                f"{m.group(2)}% of {m.group(1)} is {int(m.group(1)) * int(m.group(2)) // 100}, "
                f"so the new total is {m.group(1)} + {int(m.group(1)) * int(m.group(2)) // 100} = {normalized_answer}."
            ),
        ),
        (
            r"^A shop sells (\d+) glasses\. Each first glass in a pair costs \$(\d+), and each second glass costs (\d+)% of that price\. How many dollars does the full set cost\?$",
            lambda m: (
                f"There are {int(m.group(1)) // 2} pairs. The second glass costs "
                f"{int(m.group(2)) * int(m.group(3)) // 100}, so each pair costs "
                f"{int(m.group(2)) + (int(m.group(2)) * int(m.group(3)) // 100)} and the full set costs {normalized_answer}."
            ),
        ),
        (
            r"^A store had (\d+) \w+ on Monday and received (\d+) more on Tuesday\. The manager wrote down aisle number (\d+), but that number is not part of the count\. If (\d+) \w+ were sold afterward, how many \w+ remain\?$",
            lambda m: (
                f"Ignore aisle number {m.group(3)}. Add {m.group(1)} and {m.group(2)}, then subtract {m.group(4)} to get {normalized_answer}."
            ),
        ),
    ]
    for pattern, builder in patterns:
        match = re.match(pattern, text)
        if match:
            return builder(match)

    numbers = [int(x) for x in re.findall(r"-?\d+", text)]
    lowered = text.lower()
    if len(numbers) >= 2:
        if " divided by " in lowered and "then add" not in lowered:
            return f"Divide {numbers[0]} by {numbers[1]} to get {normalized_answer}."
        if any(token in lowered for token in [" plus ", " total", " altogether", " gets ", " bought ", " join", " added", " get on", " in total"]):
            return f"Add the needed quantities to get {normalized_answer}."
        if any(token in lowered for token in [" minus ", " remain", " left", " fly away", " gave away", " sells ", " sold ", " eaten"]):
            return f"Subtract the amount removed to get {normalized_answer}."
        if any(token in lowered for token in [" times ", " packs of", " rows with", " each row", " each pack"]):
            return f"Multiply the grouped quantities to get {normalized_answer}."

    if numeric_answer is not None:
        return f"The arithmetic result is {numeric_answer}."
    return "I compute the arithmetic result carefully."


def build_assistant_response(question: str, answer: str) -> str:
    normalized_answer = normalize_answer(answer)
    return (
        "<reasoning>\n"
        f"{build_gold_reasoning(question, answer)}\n"
        "</reasoning>\n"
        "<answer>\n"
        f"{normalized_answer}\n"
        "</answer>"
    )


# 从 GSM8K 风格的 #### 5 中抽取真实答案。
def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####", 1)[1].strip()


def samples_to_dataset(samples: list[dict[str, str]]) -> Dataset:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        hint = str(sample.get("hint", "")).strip()
        row = {
            "question": sample["question"],
            "hint": hint,
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": build_user_prompt(sample["question"], hint),
                },
            ],
            "answer": extract_hash_answer(sample["answer"]),
        }
        assistant_response = str(sample.get("assistant_response", "")).strip()
        if assistant_response:
            row["assistant_response"] = assistant_response
        rows.append(row)
    return Dataset.from_list(rows)


def normalize_question(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).lower()


def dedupe_samples(samples: list[dict[str, str]]) -> list[dict[str, str]]:
    deduped: list[dict[str, str]] = []
    seen_questions: set[str] = set()
    for sample in samples:
        key = normalize_question(sample["question"])
        if key in seen_questions:
            continue
        seen_questions.add(key)
        deduped.append(sample)
    return deduped


def assert_no_question_overlap(
    train_samples: list[dict[str, str]],
    eval_samples: list[dict[str, str]],
) -> None:
    train_questions = {normalize_question(sample["question"]) for sample in train_samples}
    overlap = [
        sample["question"]
        for sample in eval_samples
        if normalize_question(sample["question"]) in train_questions
    ]
    if overlap:
        preview = ", ".join(overlap[:3])
        raise ValueError(f"Train/eval leakage detected via overlapping questions: {preview}")


# 构建一个很小的本地数据集，避免依赖外部下载也能先把 GRPO 流程跑通。
def build_local_dataset() -> Dataset:
    samples = LOCAL_SAMPLES[:3] if SMOKE_TEST else LOCAL_SAMPLES
    return samples_to_dataset(samples)


def build_medium_synthetic_question(rng: random.Random) -> tuple[str, int]:
    people = ["John", "Mia", "Sam", "Lily", "Noah", "Emma"]
    objects = ["apples", "books", "stickers", "marbles", "cards", "cookies"]
    template_type = rng.choice(["add", "sub", "mul", "div", "word_add", "word_sub"])
    if template_type == "add":
        a = rng.randint(2, 99)
        b = rng.randint(2, 99)
        return f"What is {a} plus {b}?", a + b
    if template_type == "sub":
        a = rng.randint(10, 120)
        b = rng.randint(1, a - 1)
        return f"What is {a} minus {b}?", a - b
    if template_type == "mul":
        a = rng.randint(2, 15)
        b = rng.randint(2, 15)
        return f"What is {a} times {b}?", a * b
    if template_type == "div":
        b = rng.randint(2, 12)
        answer = rng.randint(2, 20)
        a = b * answer
        return f"What is {a} divided by {b}?", answer
    if template_type == "word_add":
        person = rng.choice(people)
        item = rng.choice(objects)
        a = rng.randint(2, 60)
        b = rng.randint(2, 40)
        question = (
            f"{person} has {a} {item} and gets {b} more. "
            f"How many {item} does {person} have now?"
        )
        return question, a + b

    person = rng.choice(people)
    item = rng.choice(objects)
    a = rng.randint(10, 70)
    b = rng.randint(1, min(30, a - 1))
    question = (
        f"{person} has {a} {item} and gives away {b}. "
        f"How many {item} does {person} have left?"
    )
    return question, a - b


def build_hard_synthetic_question(rng: random.Random) -> tuple[str, int]:
    people = ["Ava", "Ben", "Chloe", "Daniel", "Ella", "Finn"]
    objects = ["marbles", "stickers", "cards", "books", "cookies", "shells"]
    containers = ["boxes", "bags", "shelves", "drawers"]
    templates = [
        "two_step_word",
        "grouping_word",
        "mixed_expression",
        "division_then_add",
        "nested_word",
        "irrelevant_info",
        "rate_ratio_word",
        "percentage_change",
        "discount_pairs",
    ]
    if SYNTHETIC_FOCUS in templates:
        template_type = SYNTHETIC_FOCUS
    elif SYNTHETIC_PROFILE == "targeted_gsm8k":
        if TARGETED_GSM8K_SCOUT:
            weighted_templates = [
                "grouping_word",
                "grouping_word",
                "division_then_add",
                "division_then_add",
                "nested_word",
                "nested_word",
                "mixed_expression",
                "mixed_expression",
                "two_step_word",
                "two_step_word",
                "rate_ratio_word",
                "rate_ratio_word",
                "percentage_change",
                "discount_pairs",
                "irrelevant_info",
            ]
        else:
            weighted_templates = [
                "rate_ratio_word",
                "rate_ratio_word",
                "percentage_change",
                "percentage_change",
                "discount_pairs",
                "discount_pairs",
                "grouping_word",
                "division_then_add",
                "nested_word",
                "irrelevant_info",
                "mixed_expression",
                "two_step_word",
            ]
        template_type = rng.choice(weighted_templates)
    else:
        template_type = rng.choice(templates)

    if template_type == "two_step_word":
        person = rng.choice(people)
        item = rng.choice(objects)
        a = rng.randint(25, 180)
        b = rng.randint(10, 90)
        c = rng.randint(5, min(60, a + b - 1))
        question = (
            f"{person} had {a} {item}. Then {person} bought {b} more and later gave away {c}. "
            f"How many {item} does {person} have now?"
        )
        return question, a + b - c

    if template_type == "grouping_word":
        container = rng.choice(containers)
        a = rng.randint(4, 15)
        b = rng.randint(6, 18)
        c = rng.randint(10, 60)
        question = (
            f"There are {a} {container} with {b} items in each. "
            f"After packing, {c} extra items are added. How many items are there in total?"
        )
        return question, a * b + c

    if template_type == "mixed_expression":
        a = rng.randint(20, 120)
        b = rng.randint(10, 70)
        c = rng.randint(2, 9)
        d = rng.randint(5, 40)
        question = f"Compute ({a} + {b}) * {c} - {d}."
        return question, (a + b) * c - d

    if template_type == "division_then_add":
        divisor = rng.randint(3, 12)
        quotient = rng.randint(12, 40)
        addend = rng.randint(8, 90)
        dividend = divisor * quotient
        question = f"Compute {dividend} divided by {divisor}, then add {addend}."
        return question, quotient + addend

    if template_type == "nested_word":
        person = rng.choice(people)
        item = rng.choice(objects)
        per_pack = rng.randint(3, 12)
        pack_count = rng.randint(4, 14)
        gave = rng.randint(5, min(35, per_pack * pack_count - 1))
        found = rng.randint(4, 25)
        question = (
            f"{person} collected {pack_count} packs of {item} with {per_pack} in each pack. "
            f"Later {person} gave away {gave} and then found {found} more. "
            f"How many {item} does {person} have now?"
        )
        return question, pack_count * per_pack - gave + found

    if template_type == "rate_ratio_word":
        person = rng.choice(people)
        item = rng.choice(objects)
        per_group = rng.randint(3, 9)
        group_count = rng.randint(3, 6)
        extra = rng.randint(2, 12)
        question = (
            f"{person} organizes {group_count} groups of {item} with {per_group} in each group. "
            f"Then {person} adds {extra} more {item}. How many {item} are there in total?"
        )
        return question, group_count * per_group + extra

    if template_type == "percentage_change":
        base = rng.randint(8, 30) * 10
        percent = rng.choice([10, 20, 25, 50])
        if percent == 25:
            increase = base // 4
        else:
            increase = base * percent // 100
        question = (
            f"A club had {base} points. Then the score increased by {percent}%. "
            f"What is the new total number of points?"
        )
        return question, base + increase

    if template_type == "discount_pairs":
        pair_count = rng.randint(3, 8)
        full_price = rng.randint(4, 12)
        discount_percent = rng.choice([25, 50])
        discounted_price = full_price * (100 - discount_percent) // 100
        question = (
            f"A shop sells {pair_count * 2} glasses. Each first glass in a pair costs ${full_price}, "
            f"and each second glass costs {100 - discount_percent}% of that price. "
            f"How many dollars does the full set cost?"
        )
        return question, pair_count * (full_price + discounted_price)

    item = rng.choice(objects)
    a = rng.randint(40, 160)
    b = rng.randint(10, 80)
    c = rng.randint(8, min(70, a + b - 1))
    noise = rng.randint(1, 12)
    question = (
        f"A store had {a} {item} on Monday and received {b} more on Tuesday. "
        f"The manager wrote down aisle number {noise}, but that number is not part of the count. "
        f"If {c} {item} were sold afterward, how many {item} remain?"
    )
    return question, a + b - c


def build_synthetic_arithmetic_samples(target_size: int, seed: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    rows: list[dict[str, str]] = []

    while len(rows) < target_size:
        if SYNTHETIC_DIFFICULTY == "hard":
            question, answer = build_hard_synthetic_question(rng)
        else:
            question, answer = build_medium_synthetic_question(rng)
        sample = {"question": question, "answer": f"#### {answer}"}
        if TARGETED_GSM8K_SCOUT and SYNTHETIC_PROFILE == "targeted_gsm8k":
            hint = build_scout_hint(question)
            if hint:
                sample["hint"] = hint
        rows.append(sample)
        if (
            ENABLE_EQUIV_SYNTHETIC_TRANSFORMS
            and SYNTHETIC_DIFFICULTY == "hard"
            and len(rows) < target_size
        ):
            transformed_question = build_equivalent_synthetic_variant(question)
            if transformed_question and normalize_question(transformed_question) != normalize_question(question):
                transformed_sample = {
                    "question": transformed_question,
                    "answer": sample["answer"],
                }
                if "hint" in sample:
                    transformed_sample["hint"] = sample["hint"]
                rows.append(transformed_sample)

    return dedupe_samples(rows)


def build_equivalent_synthetic_variant(question: str) -> str:
    text = question.strip()
    variant_patterns: list[tuple[str, Any]] = [
        (
            r"^There are (\d+) (\w+) with (\d+) items in each\. After packing, (\d+) extra items are added\. How many items are there in total\?$",
            lambda m: (
                f"Each of the {m.group(1)} {m.group(2)} holds {m.group(3)} items, "
                f"and later {m.group(4)} extra items are packed in. "
                "How many items are there altogether?"
            ),
        ),
        (
            r"^Compute (\d+) divided by (\d+), then add (\d+)\.$",
            lambda m: (
                f"First divide {m.group(1)} by {m.group(2)}. "
                f"After that, add {m.group(3)}. What total do you get?"
            ),
        ),
        (
            r"^(\w+) organizes (\d+) groups of (\w+) with (\d+) in each group\. Then \w+ adds (\d+) more \w+\. How many \w+ are there in total\?$",
            lambda m: (
                f"{m.group(1)} sets up {m.group(2)} equal groups of {m.group(3)}, "
                f"with {m.group(4)} in every group, and then puts in {m.group(5)} more. "
                f"How many {m.group(3)} are there now?"
            ),
        ),
        (
            r"^A club had (\d+) points\. Then the score increased by (\d+)%\. What is the new total number of points\?$",
            lambda m: (
                f"A club starts with {m.group(1)} points and then gains {m.group(2)}% more points. "
                "What is the final score?"
            ),
        ),
        (
            r"^A shop sells (\d+) glasses\. Each first glass in a pair costs \$(\d+), and each second glass costs (\d+)% of that price\. How many dollars does the full set cost\?$",
            lambda m: (
                f"There are {m.group(1)} glasses arranged into pairs. In each pair, one glass costs ${m.group(2)}, "
                f"and the other costs {m.group(3)}% as much. What is the total cost of all the glasses?"
            ),
        ),
        (
            r"^(\w+) had (\d+) (\w+)\. Then \w+ bought (\d+) more and later gave away (\d+)\. How many \w+ does \w+ have now\?$",
            lambda m: (
                f"{m.group(1)} starts with {m.group(2)} {m.group(3)}, buys {m.group(4)} more, "
                f"and later gives away {m.group(5)}. How many {m.group(3)} remain?"
            ),
        ),
        (
            r"^(\w+) collected (\d+) packs of (\w+) with (\d+) in each pack\. Later \w+ gave away (\d+) and then found (\d+) more\. How many \w+ does \w+ have now\?$",
            lambda m: (
                f"{m.group(1)} has {m.group(2)} packs of {m.group(3)} with {m.group(4)} per pack. "
                f"After giving away {m.group(5)} and then finding {m.group(6)} more, "
                f"how many {m.group(3)} does {m.group(1)} have?"
            ),
        ),
        (
            r"^A store had (\d+) (\w+) on Monday and received (\d+) more on Tuesday\. The manager wrote down aisle number (\d+), but that number is not part of the count\. If (\d+) \w+ were sold afterward, how many \w+ remain\?$",
            lambda m: (
                f"A store starts with {m.group(1)} {m.group(2)} and receives {m.group(3)} more the next day. "
                f"The aisle number {m.group(4)} is irrelevant. If {m.group(5)} are sold after that, "
                f"how many {m.group(2)} are left?"
            ),
        ),
    ]
    for pattern, builder in variant_patterns:
        match = re.match(pattern, text)
        if match:
            return builder(match)
    return ""


def build_scout_hint(question: str) -> str:
    text = question.strip()
    hint_patterns: list[tuple[str, Any]] = [
        (
            r"^Compute \((\d+) \+ (\d+)\) \* (\d+) - (\d+)\.$",
            lambda _m: "Add inside the parentheses first, then multiply, then subtract.",
        ),
        (
            r"^Compute (\d+) divided by (\d+), then add (\d+)\.$",
            lambda _m: "Finish the division before adding the remaining amount.",
        ),
        (
            r"^There are (\d+) \w+ with (\d+) items in each\. After packing, (\d+) extra items are added\. How many items are there in total\?$",
            lambda _m: "Multiply the groups first, then add the extra items.",
        ),
        (
            r"^\w+ collected (\d+) packs of \w+ with (\d+) in each pack\. Later \w+ gave away (\d+) and then found (\d+) more\. How many \w+ does \w+ have now\?$",
            lambda _m: "Find the packed total first, then subtract what was given away, then add what was found.",
        ),
        (
            r"^\w+ organizes (\d+) groups of \w+ with (\d+) in each group\. Then \w+ adds (\d+) more \w+\. How many \w+ are there in total\?$",
            lambda _m: "Compute the grouped total first, then add the final extra amount.",
        ),
        (
            r"^A club had (\d+) points\. Then the score increased by (\d+)%\. What is the new total number of points\?$",
            lambda _m: "Find the increase amount first, then add it back to the original total.",
        ),
        (
            r"^A shop sells (\d+) glasses\. Each first glass in a pair costs \$(\d+), and each second glass costs (\d+)% of that price\. How many dollars does the full set cost\?$",
            lambda _m: "Work out the cost of one pair first, then multiply by the number of pairs.",
        ),
        (
            r"^A store had (\d+) \w+ on Monday and received (\d+) more on Tuesday\. The manager wrote down aisle number (\d+), but that number is not part of the count\. If (\d+) \w+ were sold afterward, how many \w+ remain\?$",
            lambda _m: "Ignore the aisle number and only combine the inventory counts before subtracting sales.",
        ),
        (
            r"^\w+ had (\d+) \w+\. Then \w+ bought (\d+) more and later gave away (\d+)\. How many \w+ does \w+ have now\?$",
            lambda _m: "Track the running total in order: add the purchase, then subtract what was given away.",
        ),
    ]
    for pattern, builder in hint_patterns:
        match = re.match(pattern, text)
        if match:
            return builder(match)
    return ""


def build_gsm8k_samples(split: str, limit: int, start_index: int = 0) -> list[dict[str, str]]:
    dataset = load_dataset(
        "openai/gsm8k",
        DATASET_CONFIG,
        split=split,
        download_config=DownloadConfig(local_files_only=HF_LOCAL_FILES_ONLY),
    )
    rows: list[dict[str, str]] = []
    start = max(0, start_index)
    stop = min(start + limit, len(dataset))
    if start >= len(dataset):
        raise ValueError(
            f"Requested GSM8K start index {start} is outside split={split} "
            f"with length={len(dataset)}."
        )
    for sample in dataset.select(range(start, stop)):
        question = str(sample["question"]).strip()
        answer = extract_hash_answer(str(sample["answer"]))
        if not question or answer is None:
            continue
        rows.append({"question": question, "answer": f"#### {answer}"})
    if not rows:
        raise ValueError(f"No usable rows were loaded from GSM8K split={split}.")
    return dedupe_samples(rows)


def resolve_synthetic_augment_count() -> int:
    synthetic_augment_count = SYNTHETIC_AUGMENT_COUNT
    if synthetic_augment_count is None:
        synthetic_augment_count = min(160, max(48, MAX_TRAIN_SAMPLES // 6))
    return max(0, synthetic_augment_count)


def resolve_anchor_replay_count(synthetic_augment_count: int) -> int:
    anchor_replay_count = ANCHOR_REPLAY_COUNT
    if anchor_replay_count is None:
        anchor_replay_count = min(64, max(16, synthetic_augment_count // 2))
    return max(0, anchor_replay_count)


def resolve_teacher_replay_count() -> int:
    teacher_replay_count = TEACHER_REPLAY_COUNT
    if teacher_replay_count is None:
        teacher_replay_count = min(48, max(12, MAX_TRAIN_SAMPLES // 20))
    return max(0, teacher_replay_count)


def resolve_dynamic_teacher_replay_count() -> int:
    dynamic_teacher_replay_count = DYNAMIC_TEACHER_REPLAY_COUNT
    if dynamic_teacher_replay_count is None:
        dynamic_teacher_replay_count = min(24, max(8, MAX_TRAIN_SAMPLES // 40))
    return max(0, dynamic_teacher_replay_count)


def resolve_prompt_replay_count() -> int:
    prompt_replay_count = PROMPT_REPLAY_COUNT
    if prompt_replay_count is None:
        prompt_replay_count = min(96, max(24, MAX_TRAIN_SAMPLES // 10))
    return max(0, prompt_replay_count)


def load_teacher_completion_bank() -> dict[str, dict[str, str]]:
    if not TEACHER_COMPLETION_BANK_PATH:
        return {}

    path = Path(TEACHER_COMPLETION_BANK_PATH)
    if not path.exists():
        print(f"[warn] teacher completion bank missing: {path}", flush=True)
        return {}

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[warn] failed to read teacher completion bank {path}: {exc}", flush=True)
        return {}

    eval_after = payload.get("eval_after", {})
    rows = eval_after.get("rows", []) if isinstance(eval_after, dict) else []
    bank: dict[str, dict[str, str]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        question = str(row.get("question", "")).strip()
        raw_completion = str(row.get("raw_completion", "")).strip()
        gold_answer = str(row.get("gold_answer", "")).strip()
        if (
            not question
            or not raw_completion
            or not gold_answer
            or not row.get("exact_match")
            or not row.get("has_strict_xml")
        ):
            continue
        bank[normalize_question(question)] = {
            "question": question,
            "answer": f"#### {gold_answer}",
            "assistant_response": raw_completion,
            "slice": classify_eval_slice(question),
        }
    return bank


def apply_teacher_completion_bank(train_samples: list[dict[str, str]]) -> list[dict[str, str]]:
    bank = load_teacher_completion_bank()
    if not bank:
        return list(train_samples)

    enhanced: list[dict[str, str]] = []
    matched = 0
    for sample in train_samples:
        enriched = dict(sample)
        teacher = bank.get(normalize_question(sample["question"]))
        if teacher is not None:
            enriched["assistant_response"] = teacher["assistant_response"]
            matched += 1
        enhanced.append(enriched)

    print(
        "[info] teacher completion bank applied: "
        f"path={TEACHER_COMPLETION_BANK_PATH}, matched={matched}/{len(train_samples)}",
        flush=True,
    )
    return enhanced


def build_teacher_replay_samples(train_samples: list[dict[str, str]]) -> list[dict[str, str]]:
    if not ENABLE_TEACHER_COMPLETION_REPLAY:
        return []

    bank = load_teacher_completion_bank()
    if not bank:
        return []

    target_size = resolve_teacher_replay_count()
    if target_size <= 0:
        return []

    prioritized: list[dict[str, str]] = []
    for sample in train_samples:
        teacher = bank.get(normalize_question(sample["question"]))
        if teacher is None:
            continue
        if TEACHER_REPLAY_SLICES and teacher["slice"] not in TEACHER_REPLAY_SLICES:
            continue
        prioritized.append(
            {
                "question": sample["question"],
                "answer": teacher["answer"],
                "hint": sample.get("hint", ""),
                "assistant_response": teacher["assistant_response"],
            }
        )
    if not prioritized:
        return []

    replay_samples: list[dict[str, str]] = []
    while len(replay_samples) < target_size:
        replay_samples.extend(prioritized[: max(1, target_size - len(replay_samples))])
    return replay_samples[:target_size]


def build_sft_teacher_bank_priority_samples(
    sft_train_samples: list[dict[str, str]],
) -> list[dict[str, str]]:
    if not ENABLE_SFT_TEACHER_BANK_PRIORITY:
        return list(sft_train_samples)

    prioritized: list[dict[str, str]] = []
    auxiliary: list[dict[str, str]] = []
    for sample in sft_train_samples:
        assistant_response = str(sample.get("assistant_response", "")).strip()
        slice_name = classify_eval_slice(sample["question"])
        if assistant_response and (
            not SFT_TEACHER_BANK_SLICES or slice_name in SFT_TEACHER_BANK_SLICES
        ):
            prioritized.append(dict(sample))
        else:
            auxiliary.append(dict(sample))

    if not prioritized:
        print(
            "[warn] SFT teacher-bank priority requested but no strict teacher rows matched; "
            "falling back to the unmodified SFT dataset",
            flush=True,
        )
        return list(sft_train_samples)

    target_size = SFT_TEACHER_BANK_TARGET
    if target_size is None:
        target_size = max(64, len(prioritized) * 4)

    keep_aux_count = SFT_TEACHER_BANK_KEEP_AUX_COUNT
    if keep_aux_count is None:
        keep_aux_count = min(len(auxiliary), max(8, len(prioritized) // 2))

    weighted: list[dict[str, str]] = []
    while len(weighted) < target_size:
        weighted.extend(dict(sample) for sample in prioritized[: max(1, target_size - len(weighted))])
    if keep_aux_count > 0:
        weighted.extend(auxiliary[:keep_aux_count])

    print(
        "[info] prioritized SFT teacher-bank mix: "
        f"teacher_rows={len(prioritized)}, "
        f"teacher_target={target_size}, "
        f"aux_rows={min(len(auxiliary), keep_aux_count)}, "
        f"combined={len(weighted)}, "
        f"slices={list(SFT_TEACHER_BANK_SLICES)}",
        flush=True,
    )
    return weighted


def build_prompt_replay_samples(train_samples: list[dict[str, str]]) -> list[dict[str, str]]:
    if not ENABLE_PROMPT_REPLAY:
        return []

    target_size = resolve_prompt_replay_count()
    if target_size <= 0:
        return []

    prioritized = [
        dict(sample)
        for sample in train_samples
        if not PROMPT_REPLAY_SLICES
        or classify_eval_slice(sample["question"]) in PROMPT_REPLAY_SLICES
    ]
    if not prioritized:
        return []

    replay_samples: list[dict[str, str]] = []
    while len(replay_samples) < target_size:
        replay_samples.extend(
            dict(sample)
            for sample in prioritized[: max(1, target_size - len(replay_samples))]
        )

    print(
        "[info] prompt replay mix ready: "
        f"source_rows={len(prioritized)}, target={target_size}, "
        f"slices={list(PROMPT_REPLAY_SLICES)}",
        flush=True,
    )
    return replay_samples[:target_size]


def build_dynamic_teacher_replay_samples(
    model,
    tokenizer,
    train_samples: list[dict[str, str]],
) -> list[dict[str, str]]:
    if not ENABLE_DYNAMIC_TEACHER_REPLAY:
        return []

    target_size = resolve_dynamic_teacher_replay_count()
    if target_size <= 0:
        return []

    prioritized = [
        sample
        for sample in train_samples
        if not DYNAMIC_TEACHER_REPLAY_SLICES
        or classify_eval_slice(sample["question"]) in DYNAMIC_TEACHER_REPLAY_SLICES
    ]
    if not prioritized:
        return []

    replay_samples: list[dict[str, str]] = []
    candidate_cap = min(len(prioritized), max(target_size * 3, target_size))
    print(
        "[info] building dynamic teacher replay: "
        f"candidate_cap={candidate_cap}, target={target_size}, "
        f"slices={list(DYNAMIC_TEACHER_REPLAY_SLICES)}",
        flush=True,
    )
    for sample in prioritized[:candidate_cap]:
        completion = generate_text(model, tokenizer, sample["question"])
        predicted_answer = normalize_answer(extract_xml_answer(completion))
        gold_answer = normalize_answer(extract_hash_answer(sample["answer"]) or "")
        if predicted_answer != gold_answer or not has_strict_xml_structure(completion):
            continue
        replay_samples.append(
            {
                "question": sample["question"],
                "answer": sample["answer"],
                "hint": sample.get("hint", ""),
                "assistant_response": completion,
            }
        )
        if len(replay_samples) >= target_size:
            break

    print(
        "[info] dynamic teacher replay ready: "
        f"selected={len(replay_samples)}/{target_size}",
        flush=True,
    )
    return replay_samples


def resolve_teacher_bank_target_count() -> int:
    teacher_bank_target = TEACHER_BANK_TARGET_COUNT
    if teacher_bank_target is None:
        teacher_bank_target = max(16, resolve_dynamic_teacher_replay_count())
    return max(1, teacher_bank_target)


def resolve_teacher_bank_candidate_cap(target_size: int, available: int) -> int:
    teacher_bank_candidate_cap = TEACHER_BANK_CANDIDATE_CAP
    if teacher_bank_candidate_cap is None:
        teacher_bank_candidate_cap = max(target_size * 4, target_size)
    return max(target_size, min(available, teacher_bank_candidate_cap))


def build_teacher_bank_payload(
    model,
    tokenizer,
    train_samples: list[dict[str, str]],
) -> dict[str, Any]:
    target_size = resolve_teacher_bank_target_count()
    prioritized = [
        sample
        for sample in train_samples
        if not TEACHER_BANK_SLICES
        or classify_eval_slice(sample["question"]) in TEACHER_BANK_SLICES
    ]
    candidate_cap = resolve_teacher_bank_candidate_cap(target_size, len(prioritized))
    selected_rows: list[dict[str, Any]] = []
    strict_exact = 0

    print(
        "[info] building teacher bank: "
        f"candidate_cap={candidate_cap}, target={target_size}, "
        f"slices={list(TEACHER_BANK_SLICES)}",
        flush=True,
    )
    for index, sample in enumerate(prioritized[:candidate_cap], start=1):
        completion = generate_text(model, tokenizer, sample["question"])
        predicted_answer = normalize_answer(extract_xml_answer(completion))
        gold_answer = normalize_answer(extract_hash_answer(sample["answer"]) or "")
        exact_match = predicted_answer == gold_answer
        answer_tag_match = has_answer_tag(completion)
        strict_xml_match = has_strict_xml_structure(completion)
        if exact_match and strict_xml_match:
            strict_exact += 1
        selected_rows.append(
            {
                "question": sample["question"],
                "gold_answer": gold_answer,
                "predicted_answer": predicted_answer,
                "exact_match": exact_match,
                "has_answer_tag": answer_tag_match,
                "has_strict_xml": strict_xml_match,
                "raw_completion": completion,
                "slice": classify_eval_slice(sample["question"]),
            }
        )
        if (
            index == 1
            or index == candidate_cap
            or index % max(1, PROGRESS_LOG_EVERY) == 0
        ):
            print(
                "[progress] teacher_bank: "
                f"{index}/{candidate_cap} (strict_exact={strict_exact})",
                flush=True,
            )
        if strict_exact >= target_size:
            break

    return {
        "eval_after": {
            "num_eval_samples": len(selected_rows),
            "rows": selected_rows,
        },
        "teacher_bank": {
            "target_count": target_size,
            "strict_exact_count": strict_exact,
            "candidate_cap": candidate_cap,
            "slices": list(TEACHER_BANK_SLICES),
        },
    }


def build_anchor_replay_samples(
    train_samples: list[dict[str, str]],
    target_size: int,
) -> list[dict[str, str]]:
    if target_size <= 0 or not ANCHOR_REPLAY_SLICES:
        return []

    prioritized: list[dict[str, str]] = []
    for slice_name in ANCHOR_REPLAY_SLICES:
        prioritized.extend(
            sample
            for sample in train_samples
            if classify_eval_slice(sample["question"]) == slice_name
        )
    if not prioritized:
        return []

    replay_samples: list[dict[str, str]] = []
    while len(replay_samples) < target_size:
        replay_samples.extend(prioritized[: max(1, target_size - len(replay_samples))])
    return replay_samples[:target_size]


def build_training_and_eval_data() -> tuple[Dataset, str, list[dict[str, str]], str, list[dict[str, str]], Dataset]:
    if DATASET_SOURCE == "local":
        train_samples = LOCAL_SAMPLES[:3] if SMOKE_TEST else LOCAL_SAMPLES
        eval_samples = dedupe_samples(LOCAL_EVAL_SAMPLES[:NUM_EVAL_SAMPLES])
        assert_no_question_overlap(train_samples, eval_samples)
        dataset = samples_to_dataset(train_samples)
        return dataset, "local", eval_samples, "local_eval_samples", train_samples, dataset
    if DATASET_SOURCE == "synthetic":
        train_target = min(MAX_TRAIN_SAMPLES, 48 if SMOKE_TEST else 512)
        eval_target = min(NUM_EVAL_SAMPLES, 8 if SMOKE_TEST else 64)
        train_samples = build_synthetic_arithmetic_samples(train_target, RANDOM_SEED)
        eval_samples = build_synthetic_arithmetic_samples(eval_target, RANDOM_SEED + 10_000)
        assert_no_question_overlap(train_samples, eval_samples)
        dataset = samples_to_dataset(train_samples)
        return dataset, "synthetic", eval_samples, "synthetic_holdout", train_samples, dataset
    if DATASET_SOURCE == "gsm8k":
        try:
            real_train_samples = build_gsm8k_samples(DATASET_SPLIT, MAX_TRAIN_SAMPLES)
            train_samples = list(real_train_samples)
            sft_train_samples = list(real_train_samples)
            synthetic_augment_count = 0
            if SYNTHETIC_DIFFICULTY == "hard":
                synthetic_augment_count = resolve_synthetic_augment_count()
                if synthetic_augment_count > 0 and not MINE_CHALLENGING_SYNTHETIC:
                    synthetic_samples = build_synthetic_arithmetic_samples(
                        synthetic_augment_count,
                        RANDOM_SEED + 30_000,
                    )
                    sft_train_samples = dedupe_samples(real_train_samples + synthetic_samples)
                    train_samples = (
                        list(real_train_samples)
                        if SYNTHETIC_SFT_ONLY
                        else list(sft_train_samples)
                    )
            if TEACHER_COMPLETION_BANK_PATH:
                sft_train_samples = apply_teacher_completion_bank(sft_train_samples)
                teacher_replay_samples = build_teacher_replay_samples(real_train_samples)
                if teacher_replay_samples:
                    sft_train_samples = list(sft_train_samples) + teacher_replay_samples
                    if ENABLE_GRPO_TEACHER_ANCHOR:
                        train_samples = list(train_samples) + teacher_replay_samples
                if ENABLE_GRPO_TEACHER_ANCHOR:
                    train_samples = apply_teacher_completion_bank(train_samples)
            sft_train_samples = build_sft_teacher_bank_priority_samples(sft_train_samples)
            prompt_replay_samples = build_prompt_replay_samples(real_train_samples)
            if prompt_replay_samples:
                train_samples = list(train_samples) + prompt_replay_samples
            if ENABLE_ANCHOR_REPLAY and ADAPTER_PATH:
                anchor_replay_count = resolve_anchor_replay_count(synthetic_augment_count)
                anchor_replay_samples = build_anchor_replay_samples(
                    train_samples[:MAX_TRAIN_SAMPLES],
                    anchor_replay_count,
                )
                if anchor_replay_samples:
                    train_samples = list(train_samples) + anchor_replay_samples
            eval_split = "test" if DATASET_SPLIT == "train" else DATASET_SPLIT
            eval_samples = build_gsm8k_samples(eval_split, NUM_EVAL_SAMPLES, DATASET_START_INDEX)
            assert_no_question_overlap(train_samples, eval_samples)
            return (
                samples_to_dataset(train_samples),
                "gsm8k",
                eval_samples,
                f"gsm8k_{eval_split}",
                train_samples,
                samples_to_dataset(sft_train_samples),
            )
        except Exception as exc:
            print(f"[warn] failed to load GSM8K, falling back to local dataset: {exc}")
            train_samples = LOCAL_SAMPLES[:3] if SMOKE_TEST else LOCAL_SAMPLES
            eval_samples = dedupe_samples(LOCAL_EVAL_SAMPLES[:NUM_EVAL_SAMPLES])
            assert_no_question_overlap(train_samples, eval_samples)
            dataset = samples_to_dataset(train_samples)
            return dataset, "local-fallback", eval_samples, "local_eval_samples", train_samples, dataset
    raise ValueError(f"Unsupported DATASET_SOURCE: {DATASET_SOURCE}")


def build_sft_dataset(source_dataset: Dataset, tokenizer) -> Dataset:
    def make_text(row: dict[str, Any]) -> dict[str, list[int]]:
        assistant_response = str(row.get("assistant_response", "")).strip()
        messages = list(row["prompt"]) + [
            {
                "role": "assistant",
                "content": assistant_response or build_assistant_response(row["question"], row["answer"]),
            }
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
        )
        encoded["labels"] = list(encoded["input_ids"])
        return encoded

    return source_dataset.map(make_text, remove_columns=source_dataset.column_names)


# 正确性奖励：如果模型输出的 <answer> 与标准答案一致，给高奖励。
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [normalize_answer(extract_xml_answer(r)) for r in responses]
    gold_answers = [normalize_answer(a) for a in answer]
    return [6.0 if r == a else 0.0 for r, a in zip(extracted_responses, gold_answers)]


def distance_reward_func(completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [normalize_answer(extract_xml_answer(r)) for r in responses]
    gold_answers = [normalize_answer(a) for a in answer]
    rewards = []

    for predicted, gold in zip(extracted_responses, gold_answers):
        if predicted == gold:
            rewards.append(0.0)
            continue
        if not re.fullmatch(r"-?\d+", predicted) or not re.fullmatch(r"-?\d+", gold):
            rewards.append(0.0)
            continue

        error = abs(int(predicted) - int(gold))
        if error == 0:
            rewards.append(0.0)
        elif error <= 2:
            rewards.append(0.5)
        elif error <= 5:
            rewards.append(0.25)
        elif error <= 10:
            rewards.append(0.1)
        else:
            rewards.append(0.0)
    return rewards


# 数字奖励：如果抽取出的答案全是数字，给一个很小的奖励，避免压过正确性奖励。
def numeric_answer_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [normalize_answer(extract_xml_answer(r)) for r in responses]
    return [0.1 if re.fullmatch(r"-?\d+", r) else 0.0 for r in extracted_responses]


# 严格格式奖励：只保留一个轻量格式奖励，避免模型只学会“包标签”。
def xml_format_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    return [xml_format_score(response) for response in responses]


def partial_credit_reward_func(completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [normalize_answer(extract_xml_answer(r)) for r in responses]
    gold_answers = [normalize_answer(a) for a in answer]
    rewards = []
    for predicted, gold in zip(extracted_responses, gold_answers):
        if predicted == gold:
            rewards.append(0.0)
        elif predicted and gold and (predicted in gold or gold in predicted):
            rewards.append(0.1)
        else:
            rewards.append(0.0)
    return rewards


def reasoning_sanity_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    rewards = []
    for response in responses:
        reasoning_match = re.search(
            r"<reasoning>\s*(.*?)\s*</reasoning>",
            response,
            flags=re.DOTALL,
        )
        if not reasoning_match:
            rewards.append(0.0)
            continue

        reasoning_text = reasoning_match.group(1).strip()
        if 3 <= len(reasoning_text.split()) <= 12:
            rewards.append(0.1)
        else:
            rewards.append(0.0)
    return rewards


def evaluate_simple_equation(equation_text: str) -> str | None:
    compact = equation_text.replace(" ", "")
    match = re.fullmatch(r"(-?\d+)([+\-*/])(-?\d+)=(-?\d+)", compact)
    if not match:
        return None

    left = int(match.group(1))
    operator = match.group(2)
    right = int(match.group(3))
    claimed = int(match.group(4))

    if operator == "+":
        actual = left + right
    elif operator == "-":
        actual = left - right
    elif operator == "*":
        actual = left * right
    else:
        if right == 0 or left % right != 0:
            return None
        actual = left // right

    return str(actual) if actual == claimed else None


def equation_consistency_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    rewards = []
    for response in responses:
        reasoning_text = extract_reasoning_text(response)
        predicted_answer = normalize_answer(extract_xml_answer(response))
        equation_candidates = re.findall(r"-?\d+\s*[+\-*/]\s*-?\d+\s*=\s*-?\d+", reasoning_text)

        if not equation_candidates:
            rewards.append(0.0)
            continue

        valid_results = [evaluate_simple_equation(candidate) for candidate in equation_candidates]
        valid_results = [result for result in valid_results if result is not None]
        invalid_count = len(equation_candidates) - len(valid_results)

        score = 0.0
        if valid_results:
            score += min(0.12, 0.06 * len(valid_results))
            if predicted_answer and predicted_answer in valid_results:
                score += 0.08
        if invalid_count > 0:
            score -= min(0.3, 0.12 * invalid_count)
            if predicted_answer and predicted_answer not in valid_results:
                score -= 0.05
        rewards.append(score)
    return rewards


def brevity_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    rewards = []
    for response in responses:
        reasoning_text = extract_reasoning_text(response)
        token_count = len(reasoning_text.split())
        if token_count == 0:
            rewards.append(-0.05)
        elif 4 <= token_count <= 18:
            rewards.append(0.05)
        elif token_count <= 30:
            rewards.append(0.0)
        else:
            rewards.append(-0.05)
    return rewards


def prompt_to_key(prompt: Any) -> str:
    return json.dumps(prompt, ensure_ascii=False, sort_keys=True)


def extract_question_from_prompt(prompt: Any) -> str:
    if isinstance(prompt, list):
        for message in reversed(prompt):
            if not isinstance(message, dict):
                continue
            if message.get("role") != "user":
                continue
            content = str(message.get("content", "")).strip()
            suffix = (
                "Answer using <reasoning> and <answer> tags. "
                "Put only the final integer inside <answer>."
            )
            if content.endswith(suffix):
                content = content[: -len(suffix)].strip()
            return content
    return str(prompt).strip()


def extract_reasoning_text(text: str) -> str:
    reasoning_match = re.search(
        r"<reasoning>\s*(.*?)\s*</reasoning>",
        text,
        flags=re.DOTALL,
    )
    if reasoning_match:
        return reasoning_match.group(1).strip()
    return ""


def normalize_text_signature(text: str) -> str:
    lowered = re.sub(r"<[^>]+>", " ", text.lower())
    lowered = re.sub(r"[^a-z0-9\s]+", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


def load_verifier_bundle() -> dict[str, Any] | None:
    global _VERIFIER_BUNDLE_CACHE, _VERIFIER_BUNDLE_CACHE_PATH, _VERIFIER_BUNDLE_ERROR
    if not VERIFIER_BUNDLE_PATH:
        return None
    if _VERIFIER_BUNDLE_CACHE is not None and _VERIFIER_BUNDLE_CACHE_PATH == VERIFIER_BUNDLE_PATH:
        return _VERIFIER_BUNDLE_CACHE
    if _VERIFIER_BUNDLE_ERROR and _VERIFIER_BUNDLE_CACHE_PATH == VERIFIER_BUNDLE_PATH:
        return None

    bundle_path = Path(VERIFIER_BUNDLE_PATH)
    try:
        with bundle_path.open("rb") as fh:
            bundle = pickle.load(fh)
    except (OSError, pickle.PickleError, AttributeError, ValueError) as exc:
        _VERIFIER_BUNDLE_CACHE = None
        _VERIFIER_BUNDLE_CACHE_PATH = VERIFIER_BUNDLE_PATH
        _VERIFIER_BUNDLE_ERROR = str(exc)
        print(f"[warn] failed to load verifier bundle from {bundle_path}: {exc}", flush=True)
        return None

    _VERIFIER_BUNDLE_CACHE = bundle
    _VERIFIER_BUNDLE_CACHE_PATH = VERIFIER_BUNDLE_PATH
    _VERIFIER_BUNDLE_ERROR = ""
    print(f"[info] loaded verifier bundle from: {bundle_path}", flush=True)
    return bundle


def verifier_metadata_tokens(
    has_answer_tag_value: bool,
    has_strict_xml_value: bool,
    is_numeric_answer_value: bool,
    completion: str,
    predicted_answer: str,
) -> str:
    length_bucket = min(len(completion) // 80, 12)
    answer_bucket = min(len(predicted_answer) // 8, 8)
    return " ".join(
        [
            f"meta_answer_tag_{int(has_answer_tag_value)}",
            f"meta_strict_xml_{int(has_strict_xml_value)}",
            f"meta_numeric_{int(is_numeric_answer_value)}",
            f"meta_completion_bucket_{length_bucket}",
            f"meta_answer_len_bucket_{answer_bucket}",
        ]
    )


def build_verifier_completion_text(question: str, completion: str, predicted_answer: str) -> str:
    normalized_question = " ".join(question.strip().split())
    normalized_completion = " ".join(completion.strip().split())
    normalized_answer = " ".join(predicted_answer.strip().split())
    meta = verifier_metadata_tokens(
        has_answer_tag(normalized_completion),
        has_strict_xml_structure(normalized_completion),
        bool(re.fullmatch(r"-?\d+", normalized_answer)),
        normalized_completion,
        normalized_answer,
    )
    return "\n".join(
        [
            f"question: {normalized_question}",
            f"predicted_answer: {normalized_answer}",
            f"completion: {normalized_completion}",
            meta,
        ]
    )


def compute_verifier_score(question: str, completion: str, predicted_answer: str) -> float:
    bundle = load_verifier_bundle()
    if bundle is None:
        return 0.0

    vectorizer = bundle.get("vectorizer")
    model = bundle.get("model")
    metrics = bundle.get("metrics", {})
    model_config = metrics.get("model", {}) if isinstance(metrics, dict) else {}
    training_mode = str(model_config.get("training_mode", "pointwise")).strip().lower()
    if vectorizer is None or model is None:
        return 0.0

    feature_text = build_verifier_completion_text(question, completion, predicted_answer)
    features = vectorizer.transform([feature_text])
    if training_mode == "pairwise":
        raw_value = float(model.decision_function(features)[0])
        return float(torch.tanh(torch.tensor(raw_value / 4.0)).item())

    probability = float(model.predict_proba(features)[0, 1])
    return probability - 0.5


def maybe_apply_verifier_tiebreak(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not VERIFIER_BUNDLE_PATH or len(candidates) < VERIFIER_MIN_CANDIDATES:
        return None

    sorted_candidates = sorted(candidates, key=lambda item: item["rerank_score"], reverse=True)
    top_score = sorted_candidates[0]["rerank_score"]
    finalists = [
        candidate
        for candidate in sorted_candidates
        if candidate["rerank_score"] >= top_score - VERIFIER_TIE_MARGIN
    ]
    finalists = [
        candidate
        for candidate in finalists
        if candidate.get("has_strict_xml", False)
        and candidate.get("has_answer_tag", False)
        and candidate.get("is_numeric_answer", False)
    ]
    if len(finalists) < VERIFIER_MIN_CANDIDATES:
        return None

    if VERIFIER_REQUIRE_ANSWER_DISAGREEMENT:
        finalist_answers = {
            candidate.get("predicted_answer", "")
            for candidate in finalists
            if candidate.get("predicted_answer", "")
        }
        if len(finalist_answers) < 2:
            return None

    return max(
        finalists,
        key=lambda item: item["rerank_score"] + VERIFIER_SCORE_WEIGHT * item.get("verifier_score", 0.0),
    )


def novelty_reward_func(prompts, completions, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    prompt_keys = [prompt_to_key(prompt) for prompt in prompts]
    grouped_indices: dict[str, list[int]] = {}
    for index, key in enumerate(prompt_keys):
        grouped_indices.setdefault(key, []).append(index)

    rewards = [0.0 for _ in responses]
    for indices in grouped_indices.values():
        signature_counts: dict[tuple[str, str], int] = {}
        for index in indices:
            predicted_answer = normalize_answer(extract_xml_answer(responses[index]))
            reasoning_signature = normalize_text_signature(extract_reasoning_text(responses[index]))
            signature = (predicted_answer, reasoning_signature)
            signature_counts[signature] = signature_counts.get(signature, 0) + 1

        for index in indices:
            predicted_answer = normalize_answer(extract_xml_answer(responses[index]))
            reasoning_signature = normalize_text_signature(extract_reasoning_text(responses[index]))
            signature = (predicted_answer, reasoning_signature)
            duplicate_count = signature_counts.get(signature, 1)
            if len(indices) > 1 and duplicate_count == 1:
                rewards[index] = 0.1
            elif duplicate_count > 1:
                rewards[index] = max(0.0, 0.04 / float(duplicate_count))
    return rewards


def infer_expected_intermediates(question: str) -> dict[str, Any] | None:
    text = question.strip()
    patterns: list[tuple[str, Any]] = [
        (
            r"^Compute \((\d+) \+ (\d+)\) \* (\d+) - (\d+)\.$",
            lambda m: {
                "intermediates": [
                    str(int(m.group(1)) + int(m.group(2))),
                    str((int(m.group(1)) + int(m.group(2))) * int(m.group(3))),
                ],
                "noise": [],
            },
        ),
        (
            r"^Compute (\d+) divided by (\d+), then add (\d+)\.$",
            lambda m: {
                "intermediates": [str(int(m.group(1)) // int(m.group(2)))],
                "noise": [],
            },
        ),
        (
            r"^There are (\d+) \w+ with (\d+) items in each\. After packing, (\d+) extra items are added\. How many items are there in total\?$",
            lambda m: {
                "intermediates": [str(int(m.group(1)) * int(m.group(2)))],
                "noise": [],
            },
        ),
        (
            r"^\w+ collected (\d+) packs of \w+ with (\d+) in each pack\. Later \w+ gave away (\d+) and then found (\d+) more\. How many \w+ does \w+ have now\?$",
            lambda m: {
                "intermediates": [
                    str(int(m.group(1)) * int(m.group(2))),
                    str(int(m.group(1)) * int(m.group(2)) - int(m.group(3))),
                ],
                "noise": [],
            },
        ),
        (
            r"^\w+ had (\d+) \w+\. Then \w+ bought (\d+) more and later gave away (\d+)\. How many \w+ does \w+ have now\?$",
            lambda m: {
                "intermediates": [str(int(m.group(1)) + int(m.group(2)))],
                "noise": [],
            },
        ),
        (
            r"^\w+ organizes (\d+) groups of \w+ with (\d+) in each group\. Then \w+ adds (\d+) more \w+\. How many \w+ are there in total\?$",
            lambda m: {
                "intermediates": [str(int(m.group(1)) * int(m.group(2)))],
                "noise": [],
            },
        ),
        (
            r"^A club had (\d+) points\. Then the score increased by (\d+)%\. What is the new total number of points\?$",
            lambda m: {
                "intermediates": [
                    str(int(m.group(1)) * int(m.group(2)) // 100),
                    str(int(m.group(1)) + (int(m.group(1)) * int(m.group(2)) // 100)),
                ],
                "noise": [],
            },
        ),
        (
            r"^A shop sells (\d+) glasses\. Each first glass in a pair costs \$(\d+), and each second glass costs (\d+)% of that price\. How many dollars does the full set cost\?$",
            lambda m: {
                "intermediates": [
                    str(int(m.group(1)) // 2),
                    str(int(m.group(2)) * int(m.group(3)) // 100),
                    str(int(m.group(2)) + (int(m.group(2)) * int(m.group(3)) // 100)),
                ],
                "noise": [],
            },
        ),
        (
            r"^A store had (\d+) \w+ on Monday and received (\d+) more on Tuesday\. The manager wrote down aisle number (\d+), but that number is not part of the count\. If (\d+) \w+ were sold afterward, how many \w+ remain\?$",
            lambda m: {
                "intermediates": [str(int(m.group(1)) + int(m.group(2)))],
                "noise": [str(int(m.group(3)))],
            },
        ),
    ]
    for pattern, builder in patterns:
        match = re.match(pattern, text)
        if match:
            return builder(match)
    return None


def step_alignment_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    gold_answers = [normalize_answer(a) for a in answer]
    rewards: list[float] = []

    for prompt, response, gold_answer in zip(prompts, responses, gold_answers):
        question = extract_question_from_prompt(prompt)
        expected = infer_expected_intermediates(question)
        if expected is None:
            rewards.append(0.0)
            continue

        reasoning_text = extract_reasoning_text(response)
        observed_numbers = {
            normalize_answer(match)
            for match in NUMBER_PATTERN.findall(reasoning_text)
        }
        predicted_answer = normalize_answer(extract_xml_answer(response))
        intermediate_hits = 0
        for value in expected["intermediates"]:
            if value in observed_numbers:
                intermediate_hits += 1

        reward = 0.0
        if intermediate_hits > 0:
            reward += 0.08 * intermediate_hits
        if expected["intermediates"] and intermediate_hits == len(expected["intermediates"]):
            reward += 0.04
        if predicted_answer == gold_answer and intermediate_hits > 0:
            reward += 0.04
        if expected["noise"] and any(noise in observed_numbers for noise in expected["noise"]):
            reward -= 0.05
        rewards.append(reward)

    return rewards


def wrong_answer_penalty_reward_func(completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    gold_answers = [normalize_answer(a) for a in answer]
    rewards: list[float] = []

    for response, gold_answer in zip(responses, gold_answers):
        predicted_answer = normalize_answer(extract_xml_answer(response))
        if predicted_answer == gold_answer:
            rewards.append(0.0)
            continue
        if not predicted_answer:
            rewards.append(-0.2)
            continue
        if not re.fullmatch(r"-?\d+", predicted_answer) or not re.fullmatch(r"-?\d+", gold_answer):
            rewards.append(-0.15)
            continue

        error = abs(int(predicted_answer) - int(gold_answer))
        if error <= 2:
            rewards.append(-0.05)
        elif error <= 5:
            rewards.append(-0.1)
        elif error <= 20:
            rewards.append(-0.2)
        else:
            rewards.append(-0.35)
    return rewards


def verifier_reward_func(prompts, completions, **kwargs) -> list[float]:
    if not VERIFIER_BUNDLE_PATH:
        return [0.0 for _ in completions]

    responses = [completion[0]["content"] for completion in completions]
    rewards: list[float] = []
    for prompt, response in zip(prompts, responses):
        question = extract_question_from_prompt(prompt)
        predicted_answer = normalize_answer(extract_xml_answer(response))
        rewards.append(compute_verifier_score(question, response, predicted_answer))
    return rewards


def teacher_anchor_reward_func(completions, assistant_response=None, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    teacher_responses = list(assistant_response) if assistant_response is not None else [""] * len(responses)
    rewards: list[float] = []

    for response, teacher_response in zip(responses, teacher_responses):
        teacher_text = str(teacher_response or "").strip()
        if not teacher_text:
            rewards.append(0.0)
            continue

        predicted_answer = normalize_answer(extract_xml_answer(response))
        teacher_answer = normalize_answer(extract_xml_answer(teacher_text))
        response_reasoning = extract_reasoning_text(response)
        teacher_reasoning = extract_reasoning_text(teacher_text)
        response_numbers = {
            normalize_answer(match)
            for match in NUMBER_PATTERN.findall(response_reasoning)
        }
        teacher_numbers = [
            normalize_answer(match)
            for match in NUMBER_PATTERN.findall(teacher_reasoning)
        ]

        reward = 0.0
        if has_answer_tag(response):
            reward += 0.04
        else:
            reward -= 0.08
        if has_strict_xml_structure(response):
            reward += 0.08
        else:
            reward -= 0.08
        if predicted_answer == teacher_answer and teacher_answer:
            reward += 0.12
        if teacher_numbers:
            matched_teacher_numbers = sum(1 for value in teacher_numbers if value in response_numbers)
            reward += min(0.08, 0.03 * matched_teacher_numbers)
            if matched_teacher_numbers == len(teacher_numbers):
                reward += 0.04
        rewards.append(reward)

    return rewards


def run_reward_parser_self_check() -> None:
    cases = [
        ("<answer>\n1,234\n</answer>", "1234"),
        ("<answer>12.0</answer>", "12"),
        ("<answer>6/3</answer>", "2"),
        ("The answer is 56", "56"),
        ("2 + 3 = 5", "5"),
        ("<answer>\n-42\n</answer>", "-42"),
    ]
    failures = []
    for raw_text, expected in cases:
        normalized = normalize_answer(extract_xml_answer(raw_text))
        if normalized != expected:
            failures.append(
                {
                    "raw_text": raw_text,
                    "expected": expected,
                    "normalized": normalized,
                }
            )

    if failures:
        raise ValueError(
            "Reward parser self-check failed:\n"
            + json.dumps(failures, ensure_ascii=False, indent=2)
        )

    print(f"[info] reward parser self-check passed ({len(cases)} cases)", flush=True)


# 加载基础模型和 tokenizer，并挂上 LoRA 适配器。
def load_model_and_tokenizer():
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        device_map="auto",
        local_files_only=HF_LOCAL_FILES_ONLY,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        local_files_only=HF_LOCAL_FILES_ONLY,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lora_config = LoraConfig(
        r=8 if SMOKE_TEST else 16,
        lora_alpha=16 if SMOKE_TEST else 32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    if ADAPTER_PATH:
        # Keep resumed adapters trainable for SFT/GRPO continuation; eval-only runs
        # still load them in frozen inference mode.
        model = PeftModel.from_pretrained(
            model,
            ADAPTER_PATH,
            is_trainable=not EVAL_ONLY,
        )
        print(f"[info] loaded adapter from: {ADAPTER_PATH}", flush=True)
    else:
        model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer


class TrainProgressCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        print(
            "[info] training started: "
            f"max_steps={state.max_steps}, "
            f"logging_steps={args.logging_steps}",
            flush=True,
        )
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return control

        parts = [f"step={state.global_step}"]
        for key in (
            "loss",
            "eval_loss",
            "reward",
            "reward_std",
            "learning_rate",
            "grad_norm",
            "epoch",
        ):
            value = logs.get(key)
            if value is None:
                continue
            if isinstance(value, float):
                parts.append(f"{key}={value:.6g}")
            else:
                parts.append(f"{key}={value}")
        print("[train] " + ", ".join(parts), flush=True)
        return control


def run_sft_warmup(model, tokenizer, source_dataset: Dataset) -> None:
    if SFT_WARMUP_STEPS <= 0:
        return

    sft_dataset = build_sft_dataset(source_dataset, tokenizer)
    sft_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "sft_warmup"),
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=1,
        learning_rate=LEARNING_RATE,
        max_steps=SFT_WARMUP_STEPS,
        logging_steps=max(1, min(5, SFT_WARMUP_STEPS)),
        save_steps=SFT_WARMUP_STEPS,
        report_to="none",
        remove_unused_columns=False,
        bf16=torch.cuda.is_available(),
    )
    trainer = Trainer(
        model=model,
        args=sft_args,
        train_dataset=sft_dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding=True,
            label_pad_token_id=-100,
        ),
    )
    trainer.add_callback(TrainProgressCallback())
    print(f"[info] starting SFT warmup for {SFT_WARMUP_STEPS} steps", flush=True)
    trainer.train()


class HeldoutEvalCallback(TrainerCallback):
    def __init__(self, tokenizer, eval_samples: list[dict[str, str]], eval_every_steps: int):
        self.tokenizer = tokenizer
        self.eval_samples = eval_samples
        self.eval_every_steps = max(1, eval_every_steps)
        self.history: list[dict[str, Any]] = []

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is None or state.global_step <= 0:
            return control
        if state.global_step % self.eval_every_steps != 0:
            return control

        metrics = evaluate_model(model, self.tokenizer, self.eval_samples)
        snapshot = {
            "step": int(state.global_step),
            "exact_match_rate": metrics["exact_match_rate"],
            "correctness_reward_mean": metrics["correctness_reward_mean"],
            "distance_reward_mean": metrics["distance_reward_mean"],
            "answer_tag_rate": metrics["answer_tag_rate"],
            "strict_xml_rate": metrics["strict_xml_rate"],
            "numeric_answer_rate": metrics["numeric_answer_rate"],
        }
        self.history.append(snapshot)
        print(
            "[info] heldout eval: "
            f"step={snapshot['step']}, "
            f"exact_match_rate={snapshot['exact_match_rate']:.2f}, "
            f"correctness_reward_mean={snapshot['correctness_reward_mean']:.2f}, "
            f"distance_reward_mean={snapshot['distance_reward_mean']:.2f}, "
            f"strict_xml_rate={snapshot['strict_xml_rate']:.2f}"
        , flush=True)
        return control


# 用当前模型做一次简单生成，方便比较训练前后效果。
def build_generation_prompt(tokenizer, user_text: str) -> str:
    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_user_prompt(user_text),
        },
    ]
    return tokenizer.apply_chat_template(
        prompt,
        tokenize=False,
        add_generation_prompt=True,
    )


def compute_candidate_confidence(model, transition_scores: torch.Tensor) -> float:
    usable_scores = transition_scores[torch.isfinite(transition_scores)]
    if usable_scores.numel() == 0:
        return float("-inf")
    return float(usable_scores.mean().item())


def summarize_transition_scores(transition_scores: torch.Tensor) -> dict[str, float]:
    usable_scores = transition_scores[torch.isfinite(transition_scores)]
    if usable_scores.numel() == 0:
        return {
            "confidence": float("-inf"),
            "min_confidence": float("-inf"),
            "low_confidence_ratio": 1.0,
        }

    probabilities = torch.exp(usable_scores)
    low_confidence_ratio = float((probabilities < LOW_CONFIDENCE_PROB_THRESHOLD).float().mean().item())
    return {
        "confidence": float(usable_scores.mean().item()),
        "min_confidence": float(usable_scores.min().item()),
        "low_confidence_ratio": low_confidence_ratio,
    }


def compute_candidate_novelty(candidate_text: str, other_texts: list[str]) -> float:
    base_tokens = set(normalize_text_signature(extract_reasoning_text(candidate_text)).split())
    if not base_tokens:
        return 0.0
    if not other_texts:
        return 1.0

    max_overlap = 0.0
    for other_text in other_texts:
        other_tokens = set(normalize_text_signature(extract_reasoning_text(other_text)).split())
        if not other_tokens:
            continue
        union = base_tokens | other_tokens
        if not union:
            continue
        overlap = len(base_tokens & other_tokens) / len(union)
        max_overlap = max(max_overlap, overlap)
    return max(0.0, 1.0 - max_overlap)


def compute_candidate_equation_support(candidate_text: str, predicted_answer: str) -> float:
    reasoning_text = extract_reasoning_text(candidate_text)
    equation_candidates = re.findall(r"-?\d+\s*[+\-*/]\s*-?\d+\s*=\s*-?\d+", reasoning_text)
    if not equation_candidates:
        return 0.0

    valid_results = [evaluate_simple_equation(candidate) for candidate in equation_candidates]
    valid_results = [result for result in valid_results if result is not None]
    invalid_count = len(equation_candidates) - len(valid_results)

    score = 0.0
    if valid_results:
        score += min(0.18, 0.08 * len(valid_results))
        if predicted_answer and predicted_answer in valid_results:
            score += 0.12
    if invalid_count > 0:
        score -= min(0.3, 0.12 * invalid_count)
        if predicted_answer and predicted_answer not in valid_results:
            score -= 0.05
    return score


def build_candidate_rows(
    model,
    tokenizer,
    inputs,
    outputs,
    user_text: str,
) -> list[dict[str, Any]]:
    prompt_length = inputs["input_ids"].shape[1]
    generated_sequences = outputs.sequences[:, prompt_length:]
    transition_scores = model.compute_transition_scores(
        outputs.sequences,
        outputs.scores,
        normalize_logits=True,
    )
    candidate_rows = []
    decoded_rows = [
        tokenizer.decode(sequence, skip_special_tokens=True).strip()
        for sequence in generated_sequences
    ]
    for decoded, token_scores in zip(decoded_rows, transition_scores):
        predicted_answer = normalize_answer(extract_xml_answer(decoded))
        confidence_stats = summarize_transition_scores(token_scores)
        candidate_rows.append(
            {
                "text": decoded,
                "predicted_answer": predicted_answer,
                "confidence": confidence_stats["confidence"],
                "min_confidence": confidence_stats["min_confidence"],
                "low_confidence_ratio": confidence_stats["low_confidence_ratio"],
                "has_answer_tag": has_answer_tag(decoded),
                "has_strict_xml": has_strict_xml_structure(decoded),
                "is_numeric_answer": bool(re.fullmatch(r"-?\d+", predicted_answer)),
                "reasoning_signature": normalize_text_signature(extract_reasoning_text(decoded)),
            }
        )

    all_texts = [row["text"] for row in candidate_rows]
    for row in candidate_rows:
        other_texts = [text for text in all_texts if text != row["text"]]
        row["novelty"] = compute_candidate_novelty(row["text"], other_texts)
        row["equation_support"] = compute_candidate_equation_support(
            row["text"],
            row["predicted_answer"],
        )
        row["verifier_score"] = compute_verifier_score(
            user_text,
            row["text"],
            row["predicted_answer"],
        )
    return candidate_rows


def rerank_candidates(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    best_single_candidate = None
    answer_counts: dict[str, int] = {}
    for candidate in candidates:
        answer_counts[candidate["predicted_answer"]] = answer_counts.get(candidate["predicted_answer"], 0) + 1

    for candidate in candidates:
        candidate["consensus_count"] = answer_counts.get(candidate["predicted_answer"], 0)
        format_bonus = 0.0
        if candidate["has_strict_xml"]:
            format_bonus += 1.0
        elif candidate["has_answer_tag"]:
            format_bonus += 0.35
        if candidate["is_numeric_answer"]:
            format_bonus += 0.25
        candidate["rerank_score"] = (
            CONFIDENCE_WEIGHT * candidate["confidence"]
            + CONSENSUS_WEIGHT * float(candidate["consensus_count"])
            + FORMAT_WEIGHT * format_bonus
            + NOVELTY_WEIGHT * candidate.get("novelty", 0.0)
            + candidate.get("equation_support", 0.0)
            - LOW_CONFIDENCE_WEIGHT * candidate.get("low_confidence_ratio", 0.0)
        )
        if (
            best_single_candidate is None
            or candidate["rerank_score"] > best_single_candidate["rerank_score"]
        ):
            best_single_candidate = candidate
    grouped_candidates: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        grouped_candidates.setdefault(candidate["predicted_answer"], []).append(candidate)

    best_answer_group: list[dict[str, Any]] | None = None
    best_answer_score = float("-inf")
    for group in grouped_candidates.values():
        group_scores = sorted(
            (candidate["rerank_score"] for candidate in group),
            reverse=True,
        )
        best_candidate_score = group_scores[0]
        support_score = (
            sum(group_scores[: min(2, len(group_scores))]) / float(min(2, len(group_scores)))
        )
        strict_count = sum(1 for candidate in group if candidate["has_strict_xml"])
        equation_support = sum(candidate.get("equation_support", 0.0) for candidate in group)
        distinct_reasoning_count = len(
            {
                candidate.get("reasoning_signature", "")
                for candidate in group
                if candidate.get("reasoning_signature", "")
            }
        )
        avg_low_confidence = sum(candidate.get("low_confidence_ratio", 0.0) for candidate in group) / float(len(group))
        group_size_bonus = ANSWER_AGG_COUNT_WEIGHT * float(max(0, len(group) - 2))
        if len(group) >= 2:
            group_size_bonus += ANSWER_AGG_PAIR_COUNT_WEIGHT
        answer_score = (
            0.6 * best_candidate_score
            + 0.4 * support_score
            + group_size_bonus
            + ANSWER_AGG_STRICT_WEIGHT * float(strict_count)
            + ANSWER_AGG_EQUATION_WEIGHT * equation_support
            + ANSWER_AGG_DIVERSITY_WEIGHT * float(max(0, distinct_reasoning_count - 1))
            - ANSWER_AGG_LOW_CONF_WEIGHT * avg_low_confidence
        )
        if answer_score > best_answer_score:
            best_answer_score = answer_score
            best_answer_group = group

    verifier_tiebreak_choice = maybe_apply_verifier_tiebreak(candidates)
    if not best_answer_group or best_single_candidate is None:
        if verifier_tiebreak_choice is not None:
            return verifier_tiebreak_choice
        return max(candidates, key=lambda item: item["rerank_score"])

    if len(best_answer_group) < ANSWER_AGG_MIN_GROUP_SIZE:
        if verifier_tiebreak_choice is not None:
            return verifier_tiebreak_choice
        return best_single_candidate

    best_group_candidate = max(best_answer_group, key=lambda item: item["rerank_score"])
    if (
        len(best_answer_group) == 2
        and best_group_candidate["predicted_answer"] != best_single_candidate["predicted_answer"]
        and (
            best_single_candidate["rerank_score"] - best_group_candidate["rerank_score"]
            > ANSWER_AGG_PAIR_MAX_SINGLE_GAP
        )
    ):
        return best_single_candidate

    if best_answer_score < best_single_candidate["rerank_score"] + ANSWER_AGG_MARGIN:
        if verifier_tiebreak_choice is not None:
            return verifier_tiebreak_choice
        return best_single_candidate

    if verifier_tiebreak_choice is not None:
        return verifier_tiebreak_choice
    return max(
        best_answer_group,
        key=lambda item: (
            item["rerank_score"]
            + 0.15 * item.get("equation_support", 0.0)
            + 0.05 * float(item.get("has_strict_xml", False))
            - 0.05 * item.get("low_confidence_ratio", 0.0)
        ),
    )


def generate_text(model, tokenizer, user_text: str) -> str:
    model.eval()
    text = build_generation_prompt(tokenizer, user_text)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    if EVAL_USE_CONFIDENCE_RERANK and EVAL_NUM_CANDIDATES > 1:
        generation_kwargs = {
            "max_new_tokens": EVAL_MAX_NEW_TOKENS,
            "do_sample": True,
            "temperature": EVAL_RERANK_TEMPERATURE,
            "top_p": EVAL_RERANK_TOP_P,
            "num_return_sequences": EVAL_NUM_CANDIDATES,
            "return_dict_in_generate": True,
            "output_scores": True,
            "pad_token_id": tokenizer.eos_token_id,
        }
        with torch.no_grad():
            outputs = model.generate(**inputs, **generation_kwargs)
        candidate_rows = build_candidate_rows(model, tokenizer, inputs, outputs, user_text)
        return rerank_candidates(candidate_rows)["text"]

    generation_kwargs = {
        "max_new_tokens": EVAL_MAX_NEW_TOKENS,
        "do_sample": False,
        "pad_token_id": tokenizer.eos_token_id,
        "temperature": None,
        "top_p": None,
        "top_k": None,
    }
    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_kwargs)
    decoded = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return decoded.strip()


def mine_challenging_synthetic_samples(
    model,
    tokenizer,
    eval_samples: list[dict[str, str]],
    target_size: int,
) -> list[dict[str, str]]:
    candidate_size = max(target_size, target_size * max(2, CANDIDATE_POOL_MULTIPLIER))
    candidate_samples = build_synthetic_arithmetic_samples(candidate_size, RANDOM_SEED + 20_000)
    assert_no_question_overlap(candidate_samples, eval_samples)

    scored_rows: list[tuple[float, dict[str, str]]] = []
    print(
        "[info] mining challenging synthetic samples: "
        f"candidate_size={len(candidate_samples)}, "
        f"log_every={PROGRESS_LOG_EVERY}",
        flush=True,
    )
    for sample in candidate_samples:
        text = build_generation_prompt(tokenizer, sample["question"])
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        generation_kwargs = {
            "max_new_tokens": 128 if SMOKE_TEST else 256,
            "do_sample": True,
            "temperature": MINE_TEMPERATURE,
            "top_p": MINE_TOP_P,
            "num_return_sequences": max(2, MINE_NUM_CANDIDATES),
            "return_dict_in_generate": True,
            "output_scores": True,
            "pad_token_id": tokenizer.eos_token_id,
        }
        with torch.no_grad():
            outputs = model.generate(**inputs, **generation_kwargs)
        candidate_rows = build_candidate_rows(model, tokenizer, inputs, outputs, sample["question"])
        best_candidate = rerank_candidates(candidate_rows)
        predicted_answer = normalize_answer(best_candidate["predicted_answer"])
        gold_answer = normalize_answer(extract_hash_answer(sample["answer"]) or "")
        answer_counts: dict[str, int] = {}
        for candidate in candidate_rows:
            answer_counts[candidate["predicted_answer"]] = answer_counts.get(candidate["predicted_answer"], 0) + 1
        disagreement_rate = 1.0 - (max(answer_counts.values()) / float(len(candidate_rows)))
        is_wrong = 1.0 if predicted_answer != gold_answer else 0.0
        difficulty_score = (
            2.0 * is_wrong
            + disagreement_rate
            + best_candidate.get("low_confidence_ratio", 0.0)
            + 0.2 * best_candidate.get("novelty", 0.0)
            - 0.1 * best_candidate.get("confidence", 0.0)
        )
        scored_rows.append((difficulty_score, sample))
        processed = len(scored_rows)
        if (
            processed == 1
            or processed == len(candidate_samples)
            or processed % max(1, PROGRESS_LOG_EVERY) == 0
        ):
            print(
                "[progress] mining: "
                f"{processed}/{len(candidate_samples)} "
                f"(current_score={difficulty_score:.3f}, disagreement_rate={disagreement_rate:.2f})",
                flush=True,
            )

    scored_rows.sort(key=lambda item: item[0], reverse=True)
    mined = [sample for _, sample in scored_rows[:target_size]]
    top_difficulty = scored_rows[0][0] if scored_rows else 0.0

    print(
        "[info] mined challenging synthetic samples: "
        f"candidate_size={len(candidate_samples)}, "
        f"selected={len(mined)}, "
        f"top_difficulty={top_difficulty:.3f}"
    , flush=True)
    return mined


def evaluate_model(
    model,
    tokenizer,
    eval_samples: list[dict[str, str]],
    *,
    phase_name: str = "eval",
) -> dict[str, Any]:
    rows = []
    exact_matches = 0
    answer_tag_matches = 0
    strict_xml_matches = 0
    numeric_matches = 0
    correctness_reward_total = 0.0
    distance_reward_total = 0.0
    abs_error_total = 0
    abs_error_count = 0
    capped_eval_samples = eval_samples[:NUM_EVAL_SAMPLES]
    print(
        f"[info] starting {phase_name}: num_samples={len(capped_eval_samples)}, "
        f"log_every={PROGRESS_LOG_EVERY}",
        flush=True,
    )
    for index, sample in enumerate(capped_eval_samples, start=1):
        completion = generate_text(model, tokenizer, sample["question"])
        predicted_answer = normalize_answer(extract_xml_answer(completion))
        gold_answer = normalize_answer(sample["answer"])
        exact_match = predicted_answer == gold_answer
        answer_tag_match = has_answer_tag(completion)
        strict_xml_match = has_strict_xml_structure(completion)
        numeric_match = bool(re.fullmatch(r"-?\d+", predicted_answer))
        correctness_reward = 6.0 if exact_match else 0.0
        distance_reward = 0.0
        abs_error = None
        if numeric_match and re.fullmatch(r"-?\d+", gold_answer):
            abs_error = abs(int(predicted_answer) - int(gold_answer))
            abs_error_total += abs_error
            abs_error_count += 1
            if not exact_match:
                if abs_error <= 2:
                    distance_reward = 0.5
                elif abs_error <= 5:
                    distance_reward = 0.25
                elif abs_error <= 10:
                    distance_reward = 0.1

        if exact_match:
            exact_matches += 1
        if answer_tag_match:
            answer_tag_matches += 1
        if strict_xml_match:
            strict_xml_matches += 1
        if numeric_match:
            numeric_matches += 1
        correctness_reward_total += correctness_reward
        distance_reward_total += distance_reward
        rows.append(
            {
                "question": sample["question"],
                "gold_answer": gold_answer,
                "predicted_answer": predicted_answer,
                "exact_match": exact_match,
                "has_answer_tag": answer_tag_match,
                "has_strict_xml": strict_xml_match,
                "is_numeric_answer": numeric_match,
                "correctness_reward": correctness_reward,
                "distance_reward": distance_reward,
                "abs_error": abs_error,
                "raw_completion": completion,
            }
        )
        if (
            index == 1
            or index == len(capped_eval_samples)
            or index % max(1, PROGRESS_LOG_EVERY) == 0
        ):
            print(
                "[progress] "
                f"{phase_name}: {index}/{len(capped_eval_samples)} "
                f"(exact_matches={exact_matches})",
                flush=True,
            )
    total = len(rows)
    slice_metrics = summarize_eval_slices(rows)
    failure_breakdown = summarize_eval_failures(rows)
    return {
        "num_eval_samples": total,
        "exact_match_count": exact_matches,
        "exact_match_rate": exact_matches / total if total else 0.0,
        "answer_tag_rate": answer_tag_matches / total if total else 0.0,
        "strict_xml_rate": strict_xml_matches / total if total else 0.0,
        "numeric_answer_rate": numeric_matches / total if total else 0.0,
        "correctness_reward_mean": correctness_reward_total / total if total else 0.0,
        "distance_reward_mean": distance_reward_total / total if total else 0.0,
        "mean_abs_error": abs_error_total / abs_error_count if abs_error_count else None,
        "slice_metrics": slice_metrics,
        "failure_breakdown": failure_breakdown,
        "rows": rows,
    }


def classify_eval_slice(question: str) -> str:
    lowered = question.lower()
    numbers = re.findall(r"-?\d+(?:\.\d+)?", question)
    if any(token in lowered for token in ["%", "percent", "discount", "interest", "raise"]):
        return "percentage"
    if any(token in lowered for token in ["each", "per", "every", "times", "double", "twice", "half"]):
        return "rate_or_ratio"
    if any(token in lowered for token in ["left", "remain", "change", "more than", "less than", "difference"]):
        return "difference"
    if len(numbers) >= 4:
        return "multi_number"
    return "basic_arithmetic"


def classify_failure_bucket(row: dict[str, Any]) -> str:
    if row.get("exact_match"):
        return "exact_match"
    if not row.get("has_answer_tag"):
        return "missing_answer_tag"
    if not row.get("has_strict_xml"):
        return "non_strict_xml"
    if not row.get("is_numeric_answer"):
        return "non_numeric_answer"

    abs_error = row.get("abs_error")
    if isinstance(abs_error, int):
        if abs_error <= 2:
            return "close_numeric_error"
        if abs_error <= 10:
            return "medium_numeric_error"
        return "large_numeric_error"
    return "other_mismatch"


def summarize_eval_slices(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, float]] = {}
    for row in rows:
        slice_name = classify_eval_slice(str(row.get("question", "")))
        bucket = grouped.setdefault(
            slice_name,
            {
                "count": 0.0,
                "exact_match_count": 0.0,
                "strict_xml_count": 0.0,
            },
        )
        bucket["count"] += 1.0
        bucket["exact_match_count"] += float(bool(row.get("exact_match")))
        bucket["strict_xml_count"] += float(bool(row.get("has_strict_xml")))

    summary: dict[str, dict[str, float]] = {}
    for slice_name, bucket in grouped.items():
        count = bucket["count"]
        summary[slice_name] = {
            "count": int(count),
            "exact_match_rate": bucket["exact_match_count"] / count if count else 0.0,
            "strict_xml_rate": bucket["strict_xml_count"] / count if count else 0.0,
        }
    return dict(sorted(summary.items()))


def summarize_eval_failures(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        bucket = classify_failure_bucket(row)
        counts[bucket] = counts.get(bucket, 0) + 1
    return dict(sorted(counts.items()))


def save_run_artifacts(
    before_text: str,
    after_text: str,
    train_result: Any,
    eval_before: dict[str, Any],
    eval_warmup: dict[str, Any] | None,
    eval_after: dict[str, Any],
    heldout_history: list[dict[str, Any]],
    dataset_name: str,
    eval_dataset_name: str,
    train_dataset_size: int,
    leakage_check_passed: bool,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metrics = {}
    if hasattr(train_result, "metrics") and isinstance(train_result.metrics, dict):
        metrics = train_result.metrics

    rerank_config = {
        "eval_use_confidence_rerank": EVAL_USE_CONFIDENCE_RERANK,
        "eval_num_candidates": EVAL_NUM_CANDIDATES,
        "eval_rerank_temperature": EVAL_RERANK_TEMPERATURE,
        "eval_rerank_top_p": EVAL_RERANK_TOP_P,
        "confidence_weight": CONFIDENCE_WEIGHT,
        "consensus_weight": CONSENSUS_WEIGHT,
        "format_weight": FORMAT_WEIGHT,
        "low_confidence_prob_threshold": LOW_CONFIDENCE_PROB_THRESHOLD,
        "low_confidence_weight": LOW_CONFIDENCE_WEIGHT,
        "novelty_weight": NOVELTY_WEIGHT,
        "answer_agg_count_weight": ANSWER_AGG_COUNT_WEIGHT,
        "answer_agg_strict_weight": ANSWER_AGG_STRICT_WEIGHT,
        "answer_agg_equation_weight": ANSWER_AGG_EQUATION_WEIGHT,
        "answer_agg_diversity_weight": ANSWER_AGG_DIVERSITY_WEIGHT,
        "answer_agg_low_conf_weight": ANSWER_AGG_LOW_CONF_WEIGHT,
        "answer_agg_min_group_size": ANSWER_AGG_MIN_GROUP_SIZE,
        "answer_agg_margin": ANSWER_AGG_MARGIN,
        "answer_agg_pair_count_weight": ANSWER_AGG_PAIR_COUNT_WEIGHT,
        "answer_agg_pair_max_single_gap": ANSWER_AGG_PAIR_MAX_SINGLE_GAP,
        "verifier_bundle_path": VERIFIER_BUNDLE_PATH,
        "verifier_score_weight": VERIFIER_SCORE_WEIGHT,
        "verifier_tie_margin": VERIFIER_TIE_MARGIN,
        "verifier_min_candidates": VERIFIER_MIN_CANDIDATES,
        "verifier_require_answer_disagreement": VERIFIER_REQUIRE_ANSWER_DISAGREEMENT,
    }
    runtime_config = {
        "random_seed": RANDOM_SEED,
        "run_mode": RUN_MODE,
        "run_protocol": RUN_PROTOCOL,
        "training_method": TRAINING_METHOD,
        "eval_only": EVAL_ONLY,
        "skip_eval_before": SKIP_EVAL_BEFORE,
        "skip_eval_warmup": SKIP_EVAL_WARMUP,
        "skip_sample_generation": SKIP_SAMPLE_GENERATION,
        "save_adapter": SAVE_ADAPTER,
        "write_explanation": WRITE_EXPLANATION,
        "adapter_path": ADAPTER_PATH,
        "continuation_safe_dynamics": CONTINUATION_SAFE_DYNAMICS,
        "synthetic_sft_only": SYNTHETIC_SFT_ONLY,
        "teacher_completion_bank_path": TEACHER_COMPLETION_BANK_PATH or None,
        "enable_teacher_completion_replay": ENABLE_TEACHER_COMPLETION_REPLAY,
        "enable_grpo_teacher_anchor": ENABLE_GRPO_TEACHER_ANCHOR,
        "teacher_replay_count": (
            resolve_teacher_replay_count() if ENABLE_TEACHER_COMPLETION_REPLAY else 0
        ),
        "teacher_replay_slices": list(TEACHER_REPLAY_SLICES),
        "enable_sft_teacher_bank_priority": ENABLE_SFT_TEACHER_BANK_PRIORITY,
        "sft_teacher_bank_target": SFT_TEACHER_BANK_TARGET,
        "sft_teacher_bank_keep_aux_count": SFT_TEACHER_BANK_KEEP_AUX_COUNT,
        "sft_teacher_bank_slices": list(SFT_TEACHER_BANK_SLICES),
        "enable_dynamic_teacher_replay": ENABLE_DYNAMIC_TEACHER_REPLAY,
        "dynamic_teacher_replay_count": (
            resolve_dynamic_teacher_replay_count() if ENABLE_DYNAMIC_TEACHER_REPLAY else 0
        ),
        "dynamic_teacher_replay_slices": list(DYNAMIC_TEACHER_REPLAY_SLICES),
        "enable_prompt_replay": ENABLE_PROMPT_REPLAY,
        "prompt_replay_count": resolve_prompt_replay_count() if ENABLE_PROMPT_REPLAY else 0,
        "prompt_replay_slices": list(PROMPT_REPLAY_SLICES),
        "enable_anchor_replay": ENABLE_ANCHOR_REPLAY,
        "anchor_replay_count": (
            resolve_anchor_replay_count(resolve_synthetic_augment_count())
            if ENABLE_ANCHOR_REPLAY
            else 0
        ),
        "anchor_replay_slices": list(ANCHOR_REPLAY_SLICES),
        "output_dir": str(OUTPUT_DIR),
        "experiment_hypothesis": EXPERIMENT_HYPOTHESIS,
        "experiment_risk": EXPERIMENT_RISK,
        "experiment_note": EXPERIMENT_NOTE,
        "grpo_dynamics": {
            "learning_rate": LEARNING_RATE,
            "beta": GRPO_BETA,
            "epsilon": GRPO_EPSILON,
            "warmup_ratio": GRPO_WARMUP_RATIO,
            "max_grad_norm": GRPO_MAX_GRAD_NORM,
            "scale_rewards": GRPO_SCALE_REWARDS,
            "loss_type": GRPO_LOSS_TYPE,
            "mask_truncated_completions": MASK_TRUNCATED_COMPLETIONS,
            "top_entropy_quantile": TOP_ENTROPY_QUANTILE,
            "off_policy_mask_threshold": OFF_POLICY_MASK_THRESHOLD,
            "use_bias_correction_kl": USE_BIAS_CORRECTION_KL,
        },
    }

    summary = {
        "smoke_test": SMOKE_TEST,
        "model_name": MODEL_NAME,
        "dataset_source": DATASET_SOURCE,
        "dataset_name": dataset_name,
        "eval_dataset_name": eval_dataset_name,
        "dataset_split": DATASET_SPLIT,
        "dataset_config": DATASET_CONFIG,
        "dataset_start_index": DATASET_START_INDEX,
        "max_train_samples": MAX_TRAIN_SAMPLES,
        "num_eval_samples": NUM_EVAL_SAMPLES,
        "train_dataset_size": train_dataset_size,
        "leakage_check_passed": leakage_check_passed,
        "max_steps": MAX_STEPS,
        "max_seq_length": MAX_SEQ_LENGTH,
        "max_prompt_length": MAX_PROMPT_LENGTH,
        "max_completion_length": MAX_COMPLETION_LENGTH,
        "num_generations": NUM_GENERATIONS,
        "per_device_train_batch_size": PER_DEVICE_TRAIN_BATCH_SIZE,
        "runtime_config": runtime_config,
        "rerank_config": rerank_config,
        "train_metrics": metrics,
        "sample_before": before_text,
        "sample_after": after_text,
        "eval_before": eval_before,
        "eval_warmup": eval_warmup,
        "eval_after": eval_after,
        "heldout_history": heldout_history,
    }

    (OUTPUT_DIR / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = f"""GRPO 运行记录
====================
smoke_test: {SMOKE_TEST}
model_name: {MODEL_NAME}
dataset_source: {DATASET_SOURCE}
dataset_name: {dataset_name}
eval_dataset_name: {eval_dataset_name}
dataset_split: {DATASET_SPLIT}
dataset_config: {DATASET_CONFIG}
dataset_start_index: {DATASET_START_INDEX}
max_train_samples: {MAX_TRAIN_SAMPLES}
num_eval_samples: {NUM_EVAL_SAMPLES}
train_dataset_size: {train_dataset_size}
leakage_check_passed: {leakage_check_passed}
max_steps: {MAX_STEPS}
max_seq_length: {MAX_SEQ_LENGTH}
max_prompt_length: {MAX_PROMPT_LENGTH}
max_completion_length: {MAX_COMPLETION_LENGTH}
num_generations: {NUM_GENERATIONS}
per_device_train_batch_size: {PER_DEVICE_TRAIN_BATCH_SIZE}

运行配置
--------------------
{json.dumps(runtime_config, ensure_ascii=False, indent=2)}

Rerank 配置
--------------------
{json.dumps(rerank_config, ensure_ascii=False, indent=2)}

训练指标
--------------------
{json.dumps(metrics, ensure_ascii=False, indent=2)}

训练前生成
--------------------
{before_text}

训练后生成
--------------------
{after_text}

训练前评测
--------------------
{json.dumps(eval_before, ensure_ascii=False, indent=2)}

SFT Warmup 后评测
--------------------
{json.dumps(eval_warmup, ensure_ascii=False, indent=2)}

训练后评测
--------------------
{json.dumps(eval_after, ensure_ascii=False, indent=2)}

Held-out 曲线
--------------------
{json.dumps(heldout_history, ensure_ascii=False, indent=2)}
"""
    (OUTPUT_DIR / "run_report.txt").write_text(report, encoding="utf-8")


def write_line_by_line_explanation() -> None:
    explanation = """llama3_1_(8b)_grpo.py 逐行解析
=================================

1-17 行：
文件头和总说明。这里明确指出脚本已经从原始 Colab Notebook 改造成适合本地执行的版本。

19-30 行：
导入依赖。核心库分别负责：
- `torch`：张量和模型执行
- `datasets`：构造训练数据集
- `peft`：给基础模型挂 LoRA
- `transformers`：加载模型和 tokenizer
- `trl`：提供 `GRPOConfig` 和 `GRPOTrainer`

33-47 行：
读取环境变量并生成默认配置。这里最关键的是默认开启 `SMOKE_TEST=1`，让脚本先以最小成本跑通。

49-56 行：
定义 system prompt，强制模型输出 `<reasoning>` 和 `<answer>` 两段，便于 reward 函数打分。

59-68 行：
两个抽取函数：
- `extract_xml_answer` 从 XML 输出里提取答案
- `extract_hash_answer` 从 `#### 5` 这种标准答案里提取数字

71-97 行：
构建本地极小训练集。每条样本都被改造成 chat 格式：
- system 消息给格式约束
- user 消息给题目
- `answer` 保存标准答案

100-143 行：
定义奖励函数，核心思路是“正确性优先，格式只做弱约束”：
- `correctness_reward_func`：答对才给主奖励
- `numeric_answer_reward_func`：答案是纯数字给很小奖励
- `xml_format_reward_func`：XML 结构正确时给轻量奖励
- `partial_credit_reward_func`：仅保留很弱的部分得分
- `reasoning_sanity_reward_func`：限制 reasoning 过短或过长

146-175 行：
加载基础模型和 tokenizer，然后通过 `LoraConfig` + `get_peft_model` 挂载 LoRA。
这里没有用原 notebook 的 `unsloth`，而是改成标准 `transformers + peft` 方案。

178-199 行：
封装一个通用推理函数 `generate_text`，用于训练前后各跑一次生成，观察行为变化。

202-245 行：
主流程前半段：
- 打印当前配置
- 构建数据集
- 加载模型
- 在训练前先生成一次样例
- 初始化 `GRPOConfig`

236-248 行：
初始化 `GRPOTrainer`。这里把 reward 函数压缩为“少量格式奖励 + 强正确性奖励”的组合。

250 行：
正式开始训练。默认 smoke test 只跑 1 step，目的是验证训练链路完整。

252-259 行：
训练结束后再次生成样例，并保存 LoRA adapter 与 tokenizer。

新增保存逻辑：
- `run_summary.json`：机器可读的运行摘要
- `run_report.txt`：人可读的运行报告
- `adapter/`：LoRA 结果

如果你要扩展成正式训练版，优先改这几项：
1. `MODEL_NAME`
2. `MAX_STEPS`
3. `MAX_SEQ_LENGTH`
4. `NUM_GENERATIONS`
5. 数据集来源
"""
    Path("/home/user/图片/llama3_1_(8b)_grpo_逐行解析.txt").write_text(
        explanation,
        encoding="utf-8",
    )


def cleanup_memory(model=None) -> None:
    if model is not None:
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def make_weighted_reward_func(reward_func, weight: float):
    if abs(weight) < 1e-12:
        def zero_reward(*args, **kwargs):
            completions = kwargs.get("completions")
            if completions is None and len(args) >= 2:
                completions = args[1]
            if completions is None:
                return []
            return [0.0 for _ in completions]
        zero_reward.__name__ = f"{reward_func.__name__}_zeroed"
        return zero_reward

    def weighted_reward(*args, **kwargs):
        values = reward_func(*args, **kwargs)
        return [weight * value for value in values]

    weighted_reward.__name__ = f"{reward_func.__name__}_x{str(weight).replace('.', '_')}"
    return weighted_reward


def build_grpo_config_kwargs() -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "output_dir": str(OUTPUT_DIR),
        "learning_rate": LEARNING_RATE,
        "adam_beta1": 0.9,
        "adam_beta2": 0.99,
        "weight_decay": 0.1,
        "warmup_ratio": GRPO_WARMUP_RATIO,
        "lr_scheduler_type": "cosine",
        "logging_steps": 1,
        "per_device_train_batch_size": PER_DEVICE_TRAIN_BATCH_SIZE,
        "gradient_accumulation_steps": 1,
        "num_generations": NUM_GENERATIONS,
        "temperature": GRPO_TEMPERATURE,
        "top_p": GRPO_TOP_P,
        "top_k": GRPO_TOP_K,
        "max_prompt_length": MAX_PROMPT_LENGTH,
        "max_completion_length": MAX_COMPLETION_LENGTH,
        "max_steps": MAX_STEPS,
        "save_steps": SAVE_STEPS,
        "max_grad_norm": GRPO_MAX_GRAD_NORM,
        "beta": GRPO_BETA,
        "epsilon": GRPO_EPSILON,
        "scale_rewards": GRPO_SCALE_REWARDS,
        "loss_type": GRPO_LOSS_TYPE,
        "mask_truncated_completions": MASK_TRUNCATED_COMPLETIONS,
        "top_entropy_quantile": TOP_ENTROPY_QUANTILE,
        "off_policy_mask_threshold": OFF_POLICY_MASK_THRESHOLD,
        "use_bias_correction_kl": USE_BIAS_CORRECTION_KL,
        "report_to": "none",
        "remove_unused_columns": False,
        "bf16": torch.cuda.is_available(),
    }
    supported = inspect.signature(GRPOConfig).parameters
    return {
        key: value
        for key, value in kwargs.items()
        if key in supported and value is not None
    }


def build_reward_funcs():
    return [
        make_weighted_reward_func(xml_format_reward_func, REWARD_WEIGHT_XML),
        make_weighted_reward_func(numeric_answer_reward_func, REWARD_WEIGHT_NUMERIC),
        make_weighted_reward_func(distance_reward_func, REWARD_WEIGHT_DISTANCE),
        make_weighted_reward_func(partial_credit_reward_func, REWARD_WEIGHT_PARTIAL),
        make_weighted_reward_func(reasoning_sanity_reward_func, REWARD_WEIGHT_REASONING),
        make_weighted_reward_func(equation_consistency_reward_func, REWARD_WEIGHT_EQUATION),
        make_weighted_reward_func(brevity_reward_func, REWARD_WEIGHT_BREVITY),
        make_weighted_reward_func(step_alignment_reward_func, REWARD_WEIGHT_STEP_ALIGN),
        make_weighted_reward_func(novelty_reward_func, REWARD_WEIGHT_NOVELTY),
        make_weighted_reward_func(wrong_answer_penalty_reward_func, REWARD_WEIGHT_WRONG_PENALTY),
        make_weighted_reward_func(verifier_reward_func, REWARD_WEIGHT_VERIFIER),
        make_weighted_reward_func(teacher_anchor_reward_func, REWARD_WEIGHT_TEACHER_ANCHOR),
        make_weighted_reward_func(correctness_reward_func, REWARD_WEIGHT_CORRECTNESS),
    ]


def extract_eval_metric(summary: dict[str, Any], key: str = "exact_match_rate") -> float:
    for section_name in ("eval_after", "eval_after_grpo", "eval_after_sft", "eval_warmup"):
        metrics = summary.get(section_name)
        if isinstance(metrics, dict):
            value = metrics.get(key)
            if isinstance(value, (int, float)):
                return float(value)
    return 0.0


def run_subprocess_attempt(label: str, env_overrides: dict[str, str], output_dir: Path) -> dict[str, Any]:
    env = os.environ.copy()
    env.update({k: str(v) for k, v in env_overrides.items()})
    env["RUN_MODE"] = "single"
    env["OUTPUT_DIR"] = str(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, str(Path(__file__).resolve())]
    print(f"[info] launching {label}: {' '.join(cmd)}", flush=True)
    print(
        "[info] overrides: "
        + json.dumps(
            {
                key: env[key]
                for key in sorted(env_overrides)
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    subprocess.run(cmd, env=env, check=True)
    summary_path = output_dir / "run_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing run summary for {label}: {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def build_auto_attempts() -> list[dict[str, Any]]:
    baseline_steps = max(SFT_BASELINE_STEPS, MAX_STEPS)

    def grpo_steps_for(total_budget: int, warmup_steps: int, requested_steps: int) -> int:
        if not FAIR_COMPARE_TOTAL_STEPS:
            return requested_steps
        return max(1, total_budget - warmup_steps)

    attempt_01_sft_steps = baseline_steps
    attempt_02_sft_steps = max(baseline_steps, max(80, MAX_STEPS))
    attempt_03_sft_steps = max(baseline_steps, max(120, MAX_STEPS))

    attempts = [
        {
            "name": "attempt_01_user_config",
            "sft_env": {
                "TRAINING_METHOD": "sft",
                "SFT_BASELINE_STEPS": str(attempt_01_sft_steps),
                "MINE_CHALLENGING_SYNTHETIC": "0",
                "EVAL_USE_CONFIDENCE_RERANK": "1",
                "EVAL_NUM_CANDIDATES": str(max(EVAL_NUM_CANDIDATES, 4)),
                "SKIP_EVAL_BEFORE": "1",
                "SKIP_SAMPLE_GENERATION": "1",
            },
            "grpo_env": {
                "TRAINING_METHOD": "grpo",
                "SFT_WARMUP_STEPS": str(SFT_WARMUP_STEPS),
                "MAX_STEPS": str(grpo_steps_for(attempt_01_sft_steps, SFT_WARMUP_STEPS, MAX_STEPS)),
                "LEARNING_RATE": str(LEARNING_RATE),
                "NUM_GENERATIONS": str(NUM_GENERATIONS),
                "PER_DEVICE_TRAIN_BATCH_SIZE": str(max(PER_DEVICE_TRAIN_BATCH_SIZE, NUM_GENERATIONS)),
                "MINE_CHALLENGING_SYNTHETIC": "0",
                "EVAL_USE_CONFIDENCE_RERANK": "1",
                "EVAL_NUM_CANDIDATES": str(max(EVAL_NUM_CANDIDATES, 4)),
                "SKIP_EVAL_BEFORE": "1",
                "SKIP_SAMPLE_GENERATION": "1",
                "DISABLE_HELDOUT_CALLBACK": "1",
            },
        },
        {
            "name": "attempt_02_grpo_warmup",
            "sft_env": {
                "TRAINING_METHOD": "sft",
                "SFT_BASELINE_STEPS": str(attempt_02_sft_steps),
                "MINE_CHALLENGING_SYNTHETIC": "0",
                "EVAL_USE_CONFIDENCE_RERANK": "1",
                "EVAL_NUM_CANDIDATES": str(max(EVAL_NUM_CANDIDATES, 6)),
                "SKIP_EVAL_BEFORE": "1",
                "SKIP_SAMPLE_GENERATION": "1",
            },
            "grpo_env": {
                "TRAINING_METHOD": "grpo",
                "SFT_WARMUP_STEPS": str(max(SFT_WARMUP_STEPS, 24)),
                "MAX_STEPS": str(
                    grpo_steps_for(attempt_02_sft_steps, max(SFT_WARMUP_STEPS, 24), max(MAX_STEPS, 140))
                ),
                "LEARNING_RATE": str(min(LEARNING_RATE, 1e-5)),
                "NUM_GENERATIONS": str(max(NUM_GENERATIONS, 4)),
                "PER_DEVICE_TRAIN_BATCH_SIZE": str(max(PER_DEVICE_TRAIN_BATCH_SIZE, 4)),
                "REWARD_WEIGHT_XML": "0.15",
                "REWARD_WEIGHT_NUMERIC": "0.05",
                "REWARD_WEIGHT_DISTANCE": "0.25",
                "REWARD_WEIGHT_PARTIAL": "0.0",
                "REWARD_WEIGHT_REASONING": "0.05",
                "REWARD_WEIGHT_EQUATION": "0.12",
                "REWARD_WEIGHT_BREVITY": "0.04",
                "REWARD_WEIGHT_NOVELTY": "0.06",
                "REWARD_WEIGHT_CORRECTNESS": "1.0",
                "MINE_CHALLENGING_SYNTHETIC": "0",
                "EVAL_USE_CONFIDENCE_RERANK": "1",
                "EVAL_NUM_CANDIDATES": str(max(EVAL_NUM_CANDIDATES, 6)),
                "LOW_CONFIDENCE_WEIGHT": "0.45",
                "NOVELTY_WEIGHT": "0.12",
                "SKIP_EVAL_BEFORE": "1",
                "SKIP_SAMPLE_GENERATION": "1",
                "DISABLE_HELDOUT_CALLBACK": "1",
            },
        },
        {
            "name": "attempt_03_grpo_stronger",
            "sft_env": {
                "TRAINING_METHOD": "sft",
                "SFT_BASELINE_STEPS": str(attempt_03_sft_steps),
                "MINE_CHALLENGING_SYNTHETIC": "0",
                "EVAL_USE_CONFIDENCE_RERANK": "1",
                "EVAL_NUM_CANDIDATES": str(max(EVAL_NUM_CANDIDATES, 8)),
                "SKIP_EVAL_BEFORE": "1",
                "SKIP_SAMPLE_GENERATION": "1",
            },
            "grpo_env": {
                "TRAINING_METHOD": "grpo",
                "SFT_WARMUP_STEPS": str(max(SFT_WARMUP_STEPS, 40)),
                "MAX_STEPS": str(
                    grpo_steps_for(attempt_03_sft_steps, max(SFT_WARMUP_STEPS, 40), max(MAX_STEPS, 180))
                ),
                "LEARNING_RATE": "8e-6",
                "NUM_GENERATIONS": str(max(NUM_GENERATIONS, 6)),
                "PER_DEVICE_TRAIN_BATCH_SIZE": str(max(PER_DEVICE_TRAIN_BATCH_SIZE, 6)),
                "REWARD_WEIGHT_XML": "0.1",
                "REWARD_WEIGHT_NUMERIC": "0.02",
                "REWARD_WEIGHT_DISTANCE": "0.2",
                "REWARD_WEIGHT_PARTIAL": "0.0",
                "REWARD_WEIGHT_REASONING": "0.02",
                "REWARD_WEIGHT_EQUATION": "0.16",
                "REWARD_WEIGHT_BREVITY": "0.06",
                "REWARD_WEIGHT_NOVELTY": "0.08",
                "REWARD_WEIGHT_CORRECTNESS": "1.25",
                "MINE_CHALLENGING_SYNTHETIC": "0",
                "EVAL_USE_CONFIDENCE_RERANK": "1",
                "EVAL_NUM_CANDIDATES": str(max(EVAL_NUM_CANDIDATES, 8)),
                "LOW_CONFIDENCE_WEIGHT": "0.55",
                "NOVELTY_WEIGHT": "0.16",
                "SKIP_EVAL_BEFORE": "1",
                "SKIP_SAMPLE_GENERATION": "1",
                "DISABLE_HELDOUT_CALLBACK": "1",
            },
        },
    ]
    return attempts[: max(1, AUTO_MAX_ATTEMPTS)]


def run_auto_improve() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    attempts = build_auto_attempts()
    overall_rows = []
    best_row = None

    for index, attempt in enumerate(attempts, start=1):
        attempt_root = OUTPUT_DIR / attempt["name"]
        sft_summary = run_subprocess_attempt(
            f"{attempt['name']} / sft",
            attempt["sft_env"],
            attempt_root / "sft_baseline",
        )
        grpo_summary = run_subprocess_attempt(
            f"{attempt['name']} / grpo",
            attempt["grpo_env"],
            attempt_root / "grpo",
        )

        sft_score = extract_eval_metric(sft_summary)
        grpo_score = extract_eval_metric(grpo_summary)
        gain = grpo_score - sft_score
        row = {
            "attempt_index": index,
            "attempt_name": attempt["name"],
            "sft_score": sft_score,
            "grpo_score": grpo_score,
            "gain_vs_sft": gain,
            "sft_output_dir": str((attempt_root / "sft_baseline").resolve()),
            "grpo_output_dir": str((attempt_root / "grpo").resolve()),
            "sft_env": attempt["sft_env"],
            "grpo_env": attempt["grpo_env"],
        }
        overall_rows.append(row)
        if best_row is None or row["gain_vs_sft"] > best_row["gain_vs_sft"]:
            best_row = row

        print(
            "[info] auto attempt result: "
            f"name={row['attempt_name']}, "
            f"sft_score={sft_score:.4f}, "
            f"grpo_score={grpo_score:.4f}, "
            f"gain_vs_sft={gain:.4f}",
            flush=True,
        )
        if gain >= MIN_GRPO_GAIN:
            print(
                "[info] stopping auto-improve because GRPO exceeded SFT "
                f"by at least {MIN_GRPO_GAIN:.4f}",
                flush=True,
            )
            break
        if not AUTO_IMPROVE_IF_NO_GAIN:
            break

    auto_summary = {
        "training_method": "auto",
        "model_name": MODEL_NAME,
        "dataset_source": DATASET_SOURCE,
        "dataset_split": DATASET_SPLIT,
        "dataset_config": DATASET_CONFIG,
        "dataset_start_index": DATASET_START_INDEX,
        "max_train_samples": MAX_TRAIN_SAMPLES,
        "num_eval_samples": NUM_EVAL_SAMPLES,
        "min_grpo_gain": MIN_GRPO_GAIN,
        "attempts": overall_rows,
        "best_attempt": best_row,
    }
    (OUTPUT_DIR / "auto_tune_summary.json").write_text(
        json.dumps(auto_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[done] auto summary saved to: {OUTPUT_DIR / 'auto_tune_summary.json'}", flush=True)


def run_single_experiment():
    seed_everything(RANDOM_SEED)
    print(f"[info] smoke_test={SMOKE_TEST}", flush=True)
    print(f"[info] model_name={MODEL_NAME}", flush=True)
    print(f"[info] max_steps={MAX_STEPS}", flush=True)
    print(f"[info] learning_rate={LEARNING_RATE}", flush=True)
    print(f"[info] random_seed={RANDOM_SEED}", flush=True)
    print(f"[info] training_method={TRAINING_METHOD}", flush=True)
    print(f"[info] run_protocol={RUN_PROTOCOL}", flush=True)
    print(f"[info] dataset_source={DATASET_SOURCE}", flush=True)
    print(f"[info] dataset_start_index={DATASET_START_INDEX}", flush=True)
    print(f"[info] max_train_samples={MAX_TRAIN_SAMPLES}", flush=True)
    print(f"[info] num_eval_samples={NUM_EVAL_SAMPLES}", flush=True)
    print(f"[info] synthetic_difficulty={SYNTHETIC_DIFFICULTY}", flush=True)
    print(f"[info] synthetic_focus={SYNTHETIC_FOCUS or 'none'}", flush=True)
    print(f"[info] eval_every_steps={EVAL_EVERY_STEPS}", flush=True)
    print(f"[info] mine_challenging_synthetic={MINE_CHALLENGING_SYNTHETIC}", flush=True)
    print(f"[info] progress_log_every={PROGRESS_LOG_EVERY}", flush=True)
    print(f"[info] sft_baseline_steps={SFT_BASELINE_STEPS}", flush=True)
    print(f"[info] skip_eval_before={SKIP_EVAL_BEFORE}", flush=True)
    print(f"[info] skip_eval_warmup={SKIP_EVAL_WARMUP}", flush=True)
    print(f"[info] skip_sample_generation={SKIP_SAMPLE_GENERATION}", flush=True)
    print(f"[info] disable_heldout_callback={DISABLE_HELDOUT_CALLBACK}", flush=True)
    print(f"[info] save_adapter={SAVE_ADAPTER}", flush=True)
    print(f"[info] write_explanation={WRITE_EXPLANATION}", flush=True)
    print(f"[info] adapter_path={ADAPTER_PATH or 'none'}", flush=True)
    print(f"[info] continuation_safe_dynamics={CONTINUATION_SAFE_DYNAMICS}", flush=True)
    print(f"[info] grpo_scale_rewards={GRPO_SCALE_REWARDS}", flush=True)
    print(f"[info] grpo_loss_type={GRPO_LOSS_TYPE}", flush=True)
    print(f"[info] mask_truncated_completions={MASK_TRUNCATED_COMPLETIONS}", flush=True)
    print(f"[info] top_entropy_quantile={TOP_ENTROPY_QUANTILE}", flush=True)
    print(f"[info] off_policy_mask_threshold={OFF_POLICY_MASK_THRESHOLD}", flush=True)
    print(f"[info] use_bias_correction_kl={USE_BIAS_CORRECTION_KL}", flush=True)
    print(f"[info] teacher_completion_bank_path={TEACHER_COMPLETION_BANK_PATH or 'none'}", flush=True)
    print(f"[info] enable_teacher_completion_replay={ENABLE_TEACHER_COMPLETION_REPLAY}", flush=True)
    print(f"[info] enable_grpo_teacher_anchor={ENABLE_GRPO_TEACHER_ANCHOR}", flush=True)
    print(f"[info] enable_dynamic_teacher_replay={ENABLE_DYNAMIC_TEACHER_REPLAY}", flush=True)
    print(f"[info] enable_prompt_replay={ENABLE_PROMPT_REPLAY}", flush=True)
    print(f"[info] build_teacher_bank_only={BUILD_TEACHER_BANK_ONLY}", flush=True)
    print(f"[info] eval_use_confidence_rerank={EVAL_USE_CONFIDENCE_RERANK}", flush=True)
    print(f"[info] eval_num_candidates={EVAL_NUM_CANDIDATES}", flush=True)
    print(f"[info] eval_only={EVAL_ONLY}", flush=True)
    print(
        "[info] grpo_dynamics="
        + json.dumps(
            {
                "learning_rate": LEARNING_RATE,
                "beta": GRPO_BETA,
                "epsilon": GRPO_EPSILON,
                "warmup_ratio": GRPO_WARMUP_RATIO,
                "max_grad_norm": GRPO_MAX_GRAD_NORM,
                "scale_rewards": GRPO_SCALE_REWARDS,
                "loss_type": GRPO_LOSS_TYPE,
                "mask_truncated_completions": MASK_TRUNCATED_COMPLETIONS,
                "top_entropy_quantile": TOP_ENTROPY_QUANTILE,
                "off_policy_mask_threshold": OFF_POLICY_MASK_THRESHOLD,
                "use_bias_correction_kl": USE_BIAS_CORRECTION_KL,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    print(
        "[info] reward_weights="
        + json.dumps(
            {
                "xml": REWARD_WEIGHT_XML,
                "numeric": REWARD_WEIGHT_NUMERIC,
                "distance": REWARD_WEIGHT_DISTANCE,
                "partial": REWARD_WEIGHT_PARTIAL,
                "reasoning": REWARD_WEIGHT_REASONING,
                "equation": REWARD_WEIGHT_EQUATION,
                "brevity": REWARD_WEIGHT_BREVITY,
                "step_align": REWARD_WEIGHT_STEP_ALIGN,
                "novelty": REWARD_WEIGHT_NOVELTY,
                "wrong_penalty": REWARD_WEIGHT_WRONG_PENALTY,
                "verifier": REWARD_WEIGHT_VERIFIER,
                "teacher_anchor": REWARD_WEIGHT_TEACHER_ANCHOR,
                "correctness": REWARD_WEIGHT_CORRECTNESS,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    print(
        "[info] rerank_weights="
        + json.dumps(
            {
                "confidence": CONFIDENCE_WEIGHT,
                "consensus": CONSENSUS_WEIGHT,
                "format": FORMAT_WEIGHT,
                "low_confidence": LOW_CONFIDENCE_WEIGHT,
                "novelty": NOVELTY_WEIGHT,
                "answer_agg_pair_count": ANSWER_AGG_PAIR_COUNT_WEIGHT,
                "answer_agg_pair_max_single_gap": ANSWER_AGG_PAIR_MAX_SINGLE_GAP,
                "temperature": EVAL_RERANK_TEMPERATURE,
                "top_p": EVAL_RERANK_TOP_P,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    print(
        "[info] verifier_config="
        + json.dumps(
            {
                "bundle_path": VERIFIER_BUNDLE_PATH or None,
                "score_weight": VERIFIER_SCORE_WEIGHT,
                "tie_margin": VERIFIER_TIE_MARGIN,
                "min_candidates": VERIFIER_MIN_CANDIDATES,
                "require_answer_disagreement": VERIFIER_REQUIRE_ANSWER_DISAGREEMENT,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    run_reward_parser_self_check()
    dataset, dataset_name, eval_samples, eval_dataset_name, train_samples, sft_source_dataset = build_training_and_eval_data()
    print(f"[info] dataset_name={dataset_name}", flush=True)
    print(f"[info] eval_dataset_name={eval_dataset_name}", flush=True)
    print(f"[info] train_dataset_size={len(dataset)}", flush=True)
    print(f"[info] eval_dataset_size={len(eval_samples)}", flush=True)
    model, tokenizer = load_model_and_tokenizer()
    if BUILD_TEACHER_BANK_ONLY:
        bank_payload = build_teacher_bank_payload(
            model,
            tokenizer,
            train_samples[:MAX_TRAIN_SAMPLES],
        )
        output_path = Path(TEACHER_BANK_OUTPUT_PATH) if TEACHER_BANK_OUTPUT_PATH else OUTPUT_DIR / "teacher_completion_bank.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(bank_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            "[done] teacher bank saved: "
            f"path={output_path}, "
            f"strict_exact_count={bank_payload['teacher_bank']['strict_exact_count']}, "
            f"num_rows={bank_payload['eval_after']['num_eval_samples']}",
            flush=True,
        )
        cleanup_memory(model)
        return
    if DATASET_SOURCE == "gsm8k" and ENABLE_DYNAMIC_TEACHER_REPLAY:
        dynamic_teacher_samples = build_dynamic_teacher_replay_samples(
            model,
            tokenizer,
            train_samples[:MAX_TRAIN_SAMPLES],
        )
        if dynamic_teacher_samples:
            sft_rows = list(sft_source_dataset.to_list())
            sft_rows.extend(samples_to_dataset(dynamic_teacher_samples).to_list())
            sft_source_dataset = Dataset.from_list(sft_rows)
            print(
                "[info] appended dynamic teacher replay samples to SFT source: "
                f"base={len(sft_rows) - len(dynamic_teacher_samples)}, "
                f"teacher={len(dynamic_teacher_samples)}, "
                f"combined={len(sft_source_dataset)}",
                flush=True,
            )
    if DATASET_SOURCE == "synthetic" and MINE_CHALLENGING_SYNTHETIC:
        train_target = len(train_samples)
        train_samples = mine_challenging_synthetic_samples(
            model,
            tokenizer,
            eval_samples,
            train_target,
        )
        assert_no_question_overlap(train_samples, eval_samples)
        dataset = samples_to_dataset(train_samples)
        print(f"[info] replaced train dataset with mined hard cases: {len(dataset)} samples", flush=True)
    elif DATASET_SOURCE == "gsm8k" and MINE_CHALLENGING_SYNTHETIC and SYNTHETIC_DIFFICULTY == "hard":
        synthetic_target = resolve_synthetic_augment_count()
        if synthetic_target > 0:
            mined_synthetic = mine_challenging_synthetic_samples(
                model,
                tokenizer,
                eval_samples,
                synthetic_target,
            )
            train_samples = dedupe_samples(train_samples + mined_synthetic)
            assert_no_question_overlap(train_samples, eval_samples)
            dataset = samples_to_dataset(train_samples)
            print(
                "[info] augmented GSM8K train dataset with mined hard synthetic cases: "
                f"real={len(train_samples) - len(mined_synthetic)}, "
                f"synthetic={len(mined_synthetic)}, "
                f"combined={len(dataset)}",
                flush=True,
            )

    if SKIP_SAMPLE_GENERATION:
        before_text = "<skipped>"
    else:
        print("\n[info] sample generation before GRPO", flush=True)
        before_text = generate_text(model, tokenizer, "Calculate 2 + 3.")
        print(before_text[:500] or "<empty>", flush=True)

    if SKIP_EVAL_BEFORE:
        eval_before = {
            "num_eval_samples": 0,
            "exact_match_count": 0,
            "exact_match_rate": 0.0,
            "answer_tag_rate": 0.0,
            "strict_xml_rate": 0.0,
            "numeric_answer_rate": 0.0,
            "correctness_reward_mean": 0.0,
            "distance_reward_mean": 0.0,
            "mean_abs_error": None,
            "rows": [],
        }
    else:
        eval_before = evaluate_model(model, tokenizer, eval_samples, phase_name="eval_before")
        print(
            "[info] eval before: "
            f"exact_match_rate={eval_before['exact_match_rate']:.2f}, "
            f"correctness_reward_mean={eval_before['correctness_reward_mean']:.2f}, "
            f"distance_reward_mean={eval_before['distance_reward_mean']:.2f}, "
            f"answer_tag_rate={eval_before['answer_tag_rate']:.2f}, "
            f"strict_xml_rate={eval_before['strict_xml_rate']:.2f}, "
            f"numeric_answer_rate={eval_before['numeric_answer_rate']:.2f}"
        , flush=True)
    eval_warmup = None

    if TRAINING_METHOD == "sft":
        original_warmup_steps = SFT_WARMUP_STEPS
        globals()["SFT_WARMUP_STEPS"] = SFT_BASELINE_STEPS
        run_sft_warmup(model, tokenizer, dataset)
        globals()["SFT_WARMUP_STEPS"] = original_warmup_steps
        if SKIP_SAMPLE_GENERATION:
            after_text = "<skipped>"
        else:
            print("\n[info] sample generation after SFT baseline", flush=True)
            after_text = generate_text(model, tokenizer, "Calculate 2 + 3.")
            print(after_text[:500] or "<empty>", flush=True)
        eval_after = evaluate_model(model, tokenizer, eval_samples, phase_name="eval_after_sft")
        print(
            "[info] eval after SFT baseline: "
            f"exact_match_rate={eval_after['exact_match_rate']:.2f}, "
            f"correctness_reward_mean={eval_after['correctness_reward_mean']:.2f}, "
            f"distance_reward_mean={eval_after['distance_reward_mean']:.2f}, "
            f"answer_tag_rate={eval_after['answer_tag_rate']:.2f}, "
            f"strict_xml_rate={eval_after['strict_xml_rate']:.2f}, "
            f"numeric_answer_rate={eval_after['numeric_answer_rate']:.2f}"
        , flush=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        if SAVE_ADAPTER:
            model.save_pretrained(OUTPUT_DIR / "adapter")
            tokenizer.save_pretrained(OUTPUT_DIR / "adapter")
        save_run_artifacts(
            before_text,
            after_text,
            None,
            eval_before,
            None,
            eval_after,
            [],
            dataset_name,
            eval_dataset_name,
            len(dataset),
            True,
        )
        cleanup_memory(model)
        if SAVE_ADAPTER:
            print(f"\n[done] saved adapter to: {OUTPUT_DIR / 'adapter'}", flush=True)
        else:
            print("\n[done] scout run completed without adapter export", flush=True)
        return

    if EVAL_ONLY:
        if SKIP_SAMPLE_GENERATION:
            after_text = "<skipped>"
        else:
            print("\n[info] sample generation in eval-only mode", flush=True)
            after_text = generate_text(model, tokenizer, "Calculate 2 + 3.")
            print(after_text[:500] or "<empty>", flush=True)
        eval_after = evaluate_model(model, tokenizer, eval_samples, phase_name="eval_only")
        print(
            "[info] eval-only metrics: "
            f"exact_match_rate={eval_after['exact_match_rate']:.2f}, "
            f"correctness_reward_mean={eval_after['correctness_reward_mean']:.2f}, "
            f"distance_reward_mean={eval_after['distance_reward_mean']:.2f}, "
            f"answer_tag_rate={eval_after['answer_tag_rate']:.2f}, "
            f"strict_xml_rate={eval_after['strict_xml_rate']:.2f}, "
            f"numeric_answer_rate={eval_after['numeric_answer_rate']:.2f}"
        , flush=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        if SAVE_ADAPTER:
            model.save_pretrained(OUTPUT_DIR / "adapter")
            tokenizer.save_pretrained(OUTPUT_DIR / "adapter")
        save_run_artifacts(
            before_text,
            after_text,
            None,
            eval_before,
            None,
            eval_after,
            [],
            dataset_name,
            eval_dataset_name,
            len(dataset),
            True,
        )
        cleanup_memory(model)
        if SAVE_ADAPTER:
            print(f"\n[done] saved adapter to: {OUTPUT_DIR / 'adapter'}", flush=True)
        else:
            print("\n[done] scout eval-only run completed without adapter export", flush=True)
        return

    run_sft_warmup(model, tokenizer, sft_source_dataset)
    if SFT_WARMUP_STEPS > 0 and not SKIP_EVAL_WARMUP:
        if not SKIP_SAMPLE_GENERATION:
            print("\n[info] sample generation after SFT warmup", flush=True)
            warmup_text = generate_text(model, tokenizer, "Calculate 2 + 3.")
            print(warmup_text[:500] or "<empty>", flush=True)
        eval_warmup = evaluate_model(model, tokenizer, eval_samples, phase_name="eval_after_sft")
        print(
            "[info] eval after SFT warmup: "
            f"exact_match_rate={eval_warmup['exact_match_rate']:.2f}, "
            f"correctness_reward_mean={eval_warmup['correctness_reward_mean']:.2f}, "
            f"distance_reward_mean={eval_warmup['distance_reward_mean']:.2f}, "
            f"answer_tag_rate={eval_warmup['answer_tag_rate']:.2f}, "
            f"strict_xml_rate={eval_warmup['strict_xml_rate']:.2f}, "
            f"numeric_answer_rate={eval_warmup['numeric_answer_rate']:.2f}"
        , flush=True)

    training_args = GRPOConfig(**build_grpo_config_kwargs())

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=build_reward_funcs(),
        args=training_args,
        train_dataset=dataset,
    )
    heldout_callback = HeldoutEvalCallback(tokenizer, eval_samples, EVAL_EVERY_STEPS)
    if not DISABLE_HELDOUT_CALLBACK:
        trainer.add_callback(heldout_callback)
    trainer.add_callback(TrainProgressCallback())

    print("[info] starting GRPO train()", flush=True)
    train_result = trainer.train()

    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True

    if SKIP_SAMPLE_GENERATION:
        after_text = "<skipped>"
    else:
        print("\n[info] sample generation after GRPO", flush=True)
        after_text = generate_text(model, tokenizer, "Calculate 2 + 3.")
        print(after_text[:500] or "<empty>", flush=True)
    eval_after = evaluate_model(model, tokenizer, eval_samples, phase_name="eval_after_grpo")
    print(
        "[info] eval after: "
        f"exact_match_rate={eval_after['exact_match_rate']:.2f}, "
        f"correctness_reward_mean={eval_after['correctness_reward_mean']:.2f}, "
        f"distance_reward_mean={eval_after['distance_reward_mean']:.2f}, "
        f"answer_tag_rate={eval_after['answer_tag_rate']:.2f}, "
        f"strict_xml_rate={eval_after['strict_xml_rate']:.2f}, "
        f"numeric_answer_rate={eval_after['numeric_answer_rate']:.2f}"
    , flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if SAVE_ADAPTER:
        model.save_pretrained(OUTPUT_DIR / "adapter")
        tokenizer.save_pretrained(OUTPUT_DIR / "adapter")
    save_run_artifacts(
        before_text,
        after_text,
        train_result,
        eval_before,
        eval_warmup,
        eval_after,
        heldout_callback.history,
        dataset_name,
        eval_dataset_name,
        len(dataset),
        True,
    )
    if WRITE_EXPLANATION:
        write_line_by_line_explanation()
    cleanup_memory(model)
    if SAVE_ADAPTER:
        print(f"\n[done] saved adapter to: {OUTPUT_DIR / 'adapter'}", flush=True)
    else:
        print("\n[done] scout GRPO run completed without adapter export", flush=True)


def main():
    if RUN_MODE == "single" or TRAINING_METHOD in {"sft", "grpo"}:
        run_single_experiment()
        return
    run_auto_improve()


if __name__ == "__main__":
    main()
