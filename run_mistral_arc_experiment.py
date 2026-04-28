from __future__ import annotations

import argparse
import gc
import inspect
import json
import os
import random
import shlex
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from datasets import concatenate_datasets, load_dataset
from huggingface_hub import HfApi, login, snapshot_download
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
from trl import SFTConfig, SFTTrainer


DEFAULT_MODEL_ID = "mistralai/Mistral-7B-v0.1"
DEFAULT_DATASET_ID = "tatsu-lab/alpaca"
DEFAULT_ARC_DATASET_ID = "allenai/ai2_arc"
DEFAULT_SCIQA_DATASET_ID = "allenai/sciq"
DEFAULT_OPENBOOKQA_DATASET_ID = "allenai/openbookqa"
DEFAULT_MODEL_NAME = "mistral-7b-qlora-alpaca-sample-0.5k"
TRAIN_COLUMNS = ["instruction", "input", "output", "text", "source"]
TRAIN_DATASET_SOURCES = {
    "alpaca": {"alpaca"},
    "arc": {"arc"},
    "sciqa": {"sciqa"},
    "openbookqa": {"openbookqa"},
    "alpaca_arc": {"alpaca", "arc"},
    "sciqa_openbookqa": {"sciqa", "openbookqa"},
    "arc_sciqa_openbookqa": {"arc", "sciqa", "openbookqa"},
    "all": {"alpaca", "arc", "sciqa", "openbookqa"},
}
TRAIN_DATASET_ALIASES = {
    "sciq": "sciqa",
    "sicq": "sciqa",
    "sicqa": "sciqa",
    "science_qa": "sciqa",
    "scienceqa": "sciqa",
    "sciq_openbookqa": "sciqa_openbookqa",
    "sicqa_openbookqa": "sciqa_openbookqa",
    "science_qa_openbookqa": "sciqa_openbookqa",
    "arc_sciq_openbookqa": "arc_sciqa_openbookqa",
    "arc_sicqa_openbookqa": "arc_sciqa_openbookqa",
}
HUB_ADAPTER_IGNORE_PATTERNS = [
    "optimizer.pt",
    "scheduler.pt",
    "rng_state*.pth",
    "scaler.pt",
    "trainer_state.json",
    "training_args.bin",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the notebook experiment: QLoRA SFT Mistral-7B on 500 "
            "Alpaca samples and evaluate ARC Challenge with lm-evaluation-harness."
        )
    )

    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--train-dataset",
        default="alpaca",
        metavar="{alpaca,arc,sciqa,openbookqa,alpaca_arc,sciqa_openbookqa,arc_sciqa_openbookqa,all}",
        help="Training data source. MCQA datasets share --arc-format prompt formatting.",
    )
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--arc-dataset-id", default=DEFAULT_ARC_DATASET_ID)
    parser.add_argument("--arc-configs", nargs="+", default=["ARC-Easy", "ARC-Challenge"])
    parser.add_argument("--arc-split", default="train")
    parser.add_argument(
        "--arc-format",
        choices=["question_answer", "question_choices_answer"],
        default="question_answer",
        help="MCQA SFT prompt format for ARC, SciQA, and OpenBookQA.",
    )
    parser.add_argument(
        "--arc-sample-size",
        type=int,
        default=0,
        help="Number of combined ARC examples to use. Use 0 or a negative value for all ARC train examples.",
    )
    parser.add_argument("--sciqa-dataset-id", "--sciq-dataset-id", dest="sciqa_dataset_id", default=DEFAULT_SCIQA_DATASET_ID)
    parser.add_argument("--sciqa-split", "--sciq-split", dest="sciqa_split", default="train")
    parser.add_argument(
        "--sciqa-sample-size",
        "--sciq-sample-size",
        dest="sciqa_sample_size",
        type=int,
        default=0,
        help="Number of SciQA examples to use. Use 0 or a negative value for all SciQA train examples.",
    )
    parser.add_argument("--openbookqa-dataset-id", default=DEFAULT_OPENBOOKQA_DATASET_ID)
    parser.add_argument("--openbookqa-configs", nargs="+", default=["main"])
    parser.add_argument("--openbookqa-split", default="train")
    parser.add_argument(
        "--openbookqa-sample-size",
        type=int,
        default=0,
        help="Number of combined OpenBookQA examples to use. Use 0 or a negative value for all examples.",
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("/home/mongjin/tmp/huggingface_cache"))
    parser.add_argument("--base-local-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--adapter-output-dir", type=Path, default=None)
    parser.add_argument("--merged-output-dir", type=Path, default=None)
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--hub-model-id", default=None)
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Pushes an adapter/model to the Hub. Use with --eval-only or --push-only.",
    )
    parser.add_argument(
        "--push-only",
        action="store_true",
        help="Push a known-good adapter/model to the Hub without running training or lm_eval.",
    )
    parser.add_argument("--max-shard-size", default="5GB")

    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=1024)

    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--save-strategy", choices=["steps", "epoch", "no"], default="steps")
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--resume-from-checkpoint", type=Path, default=None)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--optim", default="paged_adamw_32bit")
    parser.add_argument("--lr-scheduler-type", default="constant")
    parser.add_argument("--max-grad-norm", type=float, default=0.3)
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--packing", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--use-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-nested-quant", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--precision", choices=["auto", "fp16", "bf16"], default="auto")
    parser.add_argument("--bnb-4bit-compute-dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--bnb-4bit-quant-type", choices=["nf4", "fp4"], default="nf4")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attn-implementation", choices=["eager", "sdpa", "flash_attention_2"], default=None)

    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", nargs="+", default=["q_proj", "v_proj"])
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--download-base", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--save-merged-model",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also merge the best adapter into the base model and save a full model. Disabled by default.",
    )
    parser.add_argument(
        "--select-best-model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save each epoch checkpoint, run zero-shot lm_eval, and save the checkpoint with best metric.",
    )
    parser.add_argument("--best-eval-tasks", default="arc_challenge")
    parser.add_argument("--best-eval-metric", default="acc_norm")
    parser.add_argument("--best-eval-batch-size", default=None)
    parser.add_argument("--best-eval-output-dir", type=Path, default=None)
    parser.add_argument("--run-eval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--eval-baseline",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Evaluate the base Mistral model instead of the fine-tuned model.",
    )
    parser.add_argument(
        "--baseline-eval-model-id",
        default=None,
        help="Model id or local path for baseline evaluation. Defaults to the downloaded base model path when available.",
    )
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-model-id", default=None)
    parser.add_argument("--eval-peft-path", type=Path, default=None)
    parser.add_argument("--eval-output-path", type=Path, default=None)
    parser.add_argument(
        "--eval-checkpoints",
        action="store_true",
        help="With --eval-only, evaluate selected checkpoint adapters or every checkpoint under a directory.",
    )
    parser.add_argument(
        "--eval-checkpoint-paths",
        nargs="+",
        type=Path,
        default=None,
        help="Specific checkpoint adapter directories to evaluate. Use this to avoid evaluating every checkpoint.",
    )
    parser.add_argument(
        "--eval-checkpoint-dir",
        type=Path,
        default=None,
        help="Directory containing checkpoint-* adapter directories. Defaults to --output-dir or cache/results.",
    )
    parser.add_argument("--checkpoint-eval-output-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-eval-metric", default="acc_norm")
    parser.add_argument("--eval-tasks", default="arc_challenge")
    parser.add_argument("--num-fewshot", type=int, default=25)
    parser.add_argument("--eval-batch-size", default="8")
    parser.add_argument("--eval-device", default="cuda:0")
    parser.add_argument(
        "--eval-dtype",
        default="float16",
        help="dtype passed to lm_eval hf model_args. Use 'none' to omit it for compatibility.",
    )
    parser.add_argument(
        "--eval-log-samples",
        action="store_true",
        help="Save per-sample lm_eval inputs, outputs, and metrics for error analysis.",
    )

    args = parser.parse_args()
    args.train_dataset = normalize_train_dataset(args.train_dataset)
    args.cache_dir = args.cache_dir.expanduser().resolve()
    if args.base_local_dir is not None:
        args.base_local_dir = args.base_local_dir.expanduser().resolve()
    if args.output_dir is not None:
        args.output_dir = args.output_dir.expanduser().resolve()
    if args.adapter_output_dir is not None:
        args.adapter_output_dir = args.adapter_output_dir.expanduser().resolve()
    if args.merged_output_dir is not None:
        args.merged_output_dir = args.merged_output_dir.expanduser().resolve()
    if args.resume_from_checkpoint is not None:
        args.resume_from_checkpoint = args.resume_from_checkpoint.expanduser().resolve()
    if args.eval_peft_path is not None:
        args.eval_peft_path = args.eval_peft_path.expanduser().resolve()
    if args.eval_output_path is not None:
        args.eval_output_path = args.eval_output_path.expanduser().resolve()
    if args.eval_checkpoint_paths is not None:
        args.eval_checkpoint_paths = [path.expanduser().resolve() for path in args.eval_checkpoint_paths]
    if args.eval_checkpoint_dir is not None:
        args.eval_checkpoint_dir = args.eval_checkpoint_dir.expanduser().resolve()
    if args.checkpoint_eval_output_dir is not None:
        args.checkpoint_eval_output_dir = args.checkpoint_eval_output_dir.expanduser().resolve()
    if args.best_eval_output_dir is not None:
        args.best_eval_output_dir = args.best_eval_output_dir.expanduser().resolve()
    return args


def normalize_train_dataset(train_dataset: str) -> str:
    normalized = TRAIN_DATASET_ALIASES.get(train_dataset, train_dataset)
    if normalized not in TRAIN_DATASET_SOURCES:
        valid_names = ", ".join(TRAIN_DATASET_SOURCES)
        raise ValueError(f"Unsupported --train-dataset {train_dataset!r}. Use one of: {valid_names}.")
    if normalized != train_dataset:
        print(f"Normalizing --train-dataset {train_dataset!r} to {normalized!r}.")
    return normalized


def ensure_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this 7B QLoRA experiment.")

    print("GPU:", torch.cuda.get_device_name(0))
    print("Capability:", torch.cuda.get_device_capability(0))
    print("bf16 supported:", torch.cuda.is_bf16_supported())


def maybe_login(token: str | None) -> None:
    if token:
        login(token=token, add_to_git_credential=False)


def dtype_from_name(name: str) -> torch.dtype:
    try:
        return getattr(torch, name)
    except AttributeError as exc:
        raise ValueError(f"Unsupported torch dtype: {name}") from exc


def resolve_precision(args: argparse.Namespace) -> None:
    if args.precision == "auto":
        args.bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        args.fp16 = torch.cuda.is_available() and not args.bf16
    else:
        args.bf16 = args.precision == "bf16"
        args.fp16 = args.precision == "fp16"

    if args.bnb_4bit_compute_dtype == "auto":
        args.bnb_4bit_compute_dtype = "bfloat16" if args.bf16 else "float16"

    print("precision:", "bf16" if args.bf16 else "fp16" if args.fp16 else "fp32")
    print("bnb_4bit_compute_dtype:", args.bnb_4bit_compute_dtype)


def get_default_output_dir(args: argparse.Namespace) -> Path:
    return args.output_dir or (args.cache_dir / "results")


def get_base_local_dir(args: argparse.Namespace) -> Path:
    return args.base_local_dir or (args.cache_dir / args.model_id)


def looks_like_model_dir(path: Path) -> bool:
    return path.is_dir() and (path / "config.json").exists()


def get_hf_cache_snapshot_dir(args: argparse.Namespace) -> Path | None:
    repo_cache_name = f"models--{args.model_id.replace('/', '--')}"
    snapshot_roots = [
        args.cache_dir / repo_cache_name / "snapshots",
        args.cache_dir / "hub" / repo_cache_name / "snapshots",
    ]

    candidates: list[Path] = []
    for snapshot_root in snapshot_roots:
        if snapshot_root.exists():
            candidates.extend(path for path in snapshot_root.iterdir() if looks_like_model_dir(path))

    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def resolve_cached_base_model_path(args: argparse.Namespace) -> Path | None:
    local_dir = get_base_local_dir(args)
    if looks_like_model_dir(local_dir):
        return local_dir
    return get_hf_cache_snapshot_dir(args)


def get_default_merged_output_dir(args: argparse.Namespace) -> Path:
    if args.merged_output_dir:
        return args.merged_output_dir
    if args.hub_model_id:
        return args.cache_dir / args.hub_model_id
    if args.train_dataset == "arc":
        return args.cache_dir / "mistral-7b-qlora-arc-train"
    if args.train_dataset == "alpaca_arc":
        return args.cache_dir / "mistral-7b-qlora-alpaca-arc"
    if args.train_dataset != "alpaca":
        dataset_slug = args.train_dataset.replace("_", "-")
        return args.cache_dir / f"mistral-7b-qlora-{dataset_slug}"
    return args.cache_dir / DEFAULT_MODEL_NAME


def get_default_adapter_output_dir(args: argparse.Namespace) -> Path:
    if args.adapter_output_dir:
        return args.adapter_output_dir
    if args.merged_output_dir and not args.save_merged_model:
        return args.merged_output_dir
    if args.train_dataset == "arc":
        return args.cache_dir / "mistral-7b-qlora-arc-train-adapter"
    if args.train_dataset == "alpaca_arc":
        return args.cache_dir / "mistral-7b-qlora-alpaca-arc-adapter"
    if args.train_dataset != "alpaca":
        dataset_slug = args.train_dataset.replace("_", "-")
        return args.cache_dir / f"mistral-7b-qlora-{dataset_slug}-adapter"
    return args.cache_dir / f"{DEFAULT_MODEL_NAME}-adapter"


def download_model_repo(args: argparse.Namespace) -> str:
    cached_model_path = resolve_cached_base_model_path(args)
    if cached_model_path:
        print(f"Using existing local base model repository: {cached_model_path}")
        return str(cached_model_path)

    if not args.download_base:
        return args.model_id

    local_dir = get_base_local_dir(args)
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    repo_path = snapshot_download(
        repo_id=args.model_id,
        cache_dir=str(args.cache_dir),
        local_dir=str(local_dir),
        token=args.hf_token,
    )
    print(f"Base model repository is saved to: {repo_path}")
    return str(local_dir)


def format_alpaca_example(example: dict[str, Any]) -> dict[str, str]:
    instruction = example["instruction"]
    input_text = example["input"]
    output = example["output"]
    text = f"<s> {instruction} here are the inputs {input_text} \n {output} </s>"
    return {
        "instruction": instruction,
        "input": input_text,
        "output": output,
        "text": text,
        "source": "alpaca",
    }


def format_mcqa_text(
    question: str,
    answer_text: str,
    labels: list[str],
    choices: list[str],
    arc_format: str,
) -> tuple[str, str, str]:
    answer_text = " " + answer_text

    if arc_format == "question_choices_answer":
        formatted_choices = ". ".join(f"{label}. {choice_text}" for label, choice_text in zip(labels, choices))
        prompt = f"{question}\nChoices: {formatted_choices}\nAnswer:"
    else:
        prompt = f"{question}\nAnswer:"

    text = f"<s>Question: {prompt}{answer_text}</s>"
    return prompt, answer_text, text


def format_arc_example(example: dict[str, Any], arc_format: str = "question_answer") -> dict[str, str]:
    choices = example["choices"]
    labels = [str(label) for label in choices["label"]]
    texts = [str(text) for text in choices["text"]]
    answer = str(example["answerKey"])
    answer_text = texts[labels.index(answer)] if answer in labels else answer
    prompt, answer_text, text = format_mcqa_text(
        str(example["question"]),
        str(answer_text),
        labels,
        texts,
        arc_format,
    )
    return {
        "instruction": "",
        "input": prompt,
        "output": answer_text,
        "text": text,
        "source": "arc",
    }


def format_sciqa_example(
    example: dict[str, Any],
    index: int,
    arc_format: str = "question_answer",
    seed: int = 42,
) -> dict[str, str]:
    correct_answer = str(example["correct_answer"]).strip()
    choice_texts = [
        correct_answer,
        str(example["distractor1"]).strip(),
        str(example["distractor2"]).strip(),
        str(example["distractor3"]).strip(),
    ]
    rng = random.Random(f"{seed}:{index}:{example['question']}")
    rng.shuffle(choice_texts)
    labels = ["A", "B", "C", "D"]
    prompt, answer_text, text = format_mcqa_text(
        str(example["question"]),
        correct_answer,
        labels,
        choice_texts,
        arc_format,
    )
    return {
        "instruction": "",
        "input": prompt,
        "output": answer_text,
        "text": text,
        "source": "sciqa",
    }


def format_openbookqa_example(
    example: dict[str, Any],
    arc_format: str = "question_answer",
    source: str = "openbookqa",
) -> dict[str, str]:
    choices = example["choices"]
    labels = [str(label) for label in choices["label"]]
    texts = [str(text) for text in choices["text"]]
    answer = str(example["answerKey"])
    answer_text = texts[labels.index(answer)] if answer in labels else answer
    prompt, answer_text, text = format_mcqa_text(
        str(example["question_stem"]),
        str(answer_text),
        labels,
        texts,
        arc_format,
    )
    return {
        "instruction": "",
        "input": prompt,
        "output": answer_text,
        "text": text,
        "source": source,
    }


def sample_dataset(dataset, sample_size: int, seed: int, label: str):
    dataset = dataset.shuffle(seed=seed)
    if sample_size <= 0:
        print(f"Using all {len(dataset)} {label} examples.")
        return dataset

    actual_size = min(sample_size, len(dataset))
    if actual_size != sample_size:
        print(f"Requested {sample_size} {label} examples, but dataset has {len(dataset)} rows. Using {actual_size}.")
    return dataset.select(range(actual_size))


def load_alpaca_training_dataset(args: argparse.Namespace):
    dataset = load_dataset(
        args.dataset_id,
        split=args.dataset_split,
        cache_dir=str(args.cache_dir),
    )
    dataset = sample_dataset(dataset, args.sample_size, args.seed, "Alpaca")
    return dataset.map(format_alpaca_example).select_columns(TRAIN_COLUMNS)


def load_arc_training_dataset(args: argparse.Namespace):
    arc_datasets = []
    for config in args.arc_configs:
        dataset = load_dataset(
            args.arc_dataset_id,
            config,
            split=args.arc_split,
            cache_dir=str(args.cache_dir),
        )
        dataset = dataset.map(format_arc_example, fn_kwargs={"arc_format": args.arc_format}).select_columns(TRAIN_COLUMNS)
        dataset = dataset.remove_columns("source").add_column("source", [config] * len(dataset))
        arc_datasets.append(dataset)
        print(f"Loaded {len(dataset)} examples from {args.arc_dataset_id}/{config}/{args.arc_split}.")

    dataset = concatenate_datasets(arc_datasets)
    dataset = sample_dataset(dataset, args.arc_sample_size, args.seed, "ARC")
    return dataset


def load_sciqa_training_dataset(args: argparse.Namespace):
    dataset = load_dataset(
        args.sciqa_dataset_id,
        split=args.sciqa_split,
        cache_dir=str(args.cache_dir),
    )
    dataset = dataset.map(
        format_sciqa_example,
        with_indices=True,
        fn_kwargs={"arc_format": args.arc_format, "seed": args.seed},
    ).select_columns(TRAIN_COLUMNS)
    dataset = sample_dataset(dataset, args.sciqa_sample_size, args.seed, "SciQA")
    return dataset


def load_openbookqa_training_dataset(args: argparse.Namespace):
    openbookqa_datasets = []
    for config in args.openbookqa_configs:
        dataset = load_dataset(
            args.openbookqa_dataset_id,
            config,
            split=args.openbookqa_split,
            cache_dir=str(args.cache_dir),
        )
        dataset = dataset.map(
            format_openbookqa_example,
            fn_kwargs={"arc_format": args.arc_format, "source": f"openbookqa-{config}"},
        ).select_columns(TRAIN_COLUMNS)
        openbookqa_datasets.append(dataset)
        print(f"Loaded {len(dataset)} examples from {args.openbookqa_dataset_id}/{config}/{args.openbookqa_split}.")

    dataset = concatenate_datasets(openbookqa_datasets)
    dataset = sample_dataset(dataset, args.openbookqa_sample_size, args.seed, "OpenBookQA")
    return dataset


def get_training_sources(train_dataset: str) -> set[str]:
    return TRAIN_DATASET_SOURCES[train_dataset]


def load_training_dataset(args: argparse.Namespace):
    sources = get_training_sources(args.train_dataset)
    datasets = []
    if "alpaca" in sources:
        datasets.append(load_alpaca_training_dataset(args))
    if "arc" in sources:
        datasets.append(load_arc_training_dataset(args))
    if "sciqa" in sources:
        datasets.append(load_sciqa_training_dataset(args))
    if "openbookqa" in sources:
        datasets.append(load_openbookqa_training_dataset(args))

    if len(datasets) == 1:
        dataset = datasets[0]
    else:
        dataset = concatenate_datasets(datasets).shuffle(seed=args.seed)

    print(dataset)
    print("First formatted sample:")
    print(dataset[0]["text"])
    return dataset


def load_model_and_tokenizer(args: argparse.Namespace, model_path: str):
    compute_dtype = dtype_from_name(args.bnb_4bit_compute_dtype)
    bnb_config = None
    if args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.use_nested_quant,
        )

    if compute_dtype == torch.float16 and args.use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16; pass --precision bf16 to use it.")
            print("=" * 80)

    model_kwargs: dict[str, Any] = {
        "device_map": args.device_map,
        "torch_dtype": compute_dtype,
        "cache_dir": str(args.cache_dir),
        "trust_remote_code": args.trust_remote_code,
    }
    if bnb_config is not None:
        model_kwargs["quantization_config"] = bnb_config
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        add_eos_token=True,
        cache_dir=str(args.cache_dir),
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer.padding_side = "right"
    set_pad_token(model, tokenizer)
    return model, tokenizer


def set_pad_token(model, tokenizer) -> None:
    if "<pad>" in tokenizer.get_vocab():
        print("<pad> token is in the tokenizer. Using <pad> for pad.")
        tokenizer.pad_token = "<pad>"
    elif "<unk>" in tokenizer.get_vocab():
        print("<unk> token is in the tokenizer. Using <unk> for pad.")
        tokenizer.pad_token = "<unk>"
    else:
        print(f"Using EOS token, {tokenizer.eos_token}, for padding.")
        tokenizer.pad_token = tokenizer.eos_token

    model.pad_token_id = tokenizer.pad_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    assert model.pad_token_id == tokenizer.pad_token_id

    print("Tokenizer pad token ID:", tokenizer.pad_token_id)
    print("Model pad token ID:", model.pad_token_id)
    print("Model config pad token ID:", model.config.pad_token_id)
    print("Number of tokens now in tokenizer:", tokenizer.vocab_size)


def print_trainable_parameters(model) -> None:
    trainable_params = 0
    total_params = 0
    for _, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    trainable_percent = 100 * trainable_params / total_params
    print(f"Trainable Parameters: {trainable_params}")
    print(f"Total Parameters: {total_params}")
    print(f"Trainable %: {trainable_percent:.2f}")


def apply_lora(args: argparse.Namespace, model):
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    print_trainable_parameters(model)
    return model


def print_versions() -> None:
    import accelerate
    import datasets
    import peft
    import transformers
    import trl

    print("trl", trl.__version__)
    print("transformers", transformers.__version__)
    print("peft", peft.__version__)
    print("accelerate", accelerate.__version__)
    print("datasets", datasets.__version__)
    print("torch", torch.__version__)


def make_sft_config(args: argparse.Namespace) -> SFTConfig:
    output_dir = get_default_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "optim": args.optim,
        "save_strategy": "epoch" if args.select_best_model else args.save_strategy,
        "save_steps": args.save_steps,
        "logging_steps": args.logging_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "bf16": args.bf16,
        "fp16": args.fp16,
        "max_grad_norm": args.max_grad_norm,
        "max_steps": -1,
        "warmup_ratio": args.warmup_ratio,
        "lr_scheduler_type": args.lr_scheduler_type,
        "dataset_text_field": "text",
        "packing": args.packing,
        "report_to": args.report_to,
    }

    fields = getattr(SFTConfig, "__dataclass_fields__", {})
    if "max_length" in fields:
        config_kwargs["max_length"] = args.max_length
    elif "max_seq_length" in fields:
        config_kwargs["max_seq_length"] = args.max_length
    else:
        config_kwargs["max_length"] = args.max_length

    return SFTConfig(**config_kwargs)


def make_trainer(model, tokenizer, dataset, training_args: SFTConfig) -> SFTTrainer:
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "train_dataset": dataset,
        "args": training_args,
    }
    signature = inspect.signature(SFTTrainer.__init__)
    if "processing_class" in signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    return SFTTrainer(**trainer_kwargs)


def print_training_state(trainer: SFTTrainer) -> None:
    print("args fp16:", trainer.args.fp16)
    print("args bf16:", trainer.args.bf16)
    try:
        print("accelerate mixed_precision:", trainer.accelerator.state.mixed_precision)
        print("scaler:", trainer.accelerator.scaler)
    except Exception as exc:
        print(exc)

    counter: Counter[str] = Counter()
    for name, param in trainer.model.named_parameters():
        if param.requires_grad:
            counter[str(param.dtype)] += param.numel()
            print(name, param.dtype, param.shape)
    print(counter)


def list_checkpoint_dirs(output_dir: Path) -> list[Path]:
    def checkpoint_step(path: Path) -> int:
        try:
            return int(path.name.rsplit("-", 1)[1])
        except (IndexError, ValueError):
            return -1

    return sorted(
        [path for path in output_dir.glob("checkpoint-*") if path.is_dir()],
        key=checkpoint_step,
    )


def get_checkpoint_epoch(checkpoint_dir: Path) -> float | None:
    state_path = checkpoint_dir / "trainer_state.json"
    if not state_path.exists():
        return None

    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None

    epoch = state.get("epoch")
    return float(epoch) if epoch is not None else None


def clean_model_output_dir(output_dir: Path) -> None:
    """Remove stale model artifacts before save_pretrained writes a new model."""
    patterns = [
        "*.safetensors",
        "*.bin",
        "*.index.json",
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "adapter_config.json",
    ]
    for pattern in patterns:
        for path in output_dir.glob(pattern):
            if path.is_file():
                path.unlink()


def push_folder_to_hub(
    args: argparse.Namespace,
    folder_path: Path,
    description: str,
    ignore_patterns: list[str] | None = None,
) -> None:
    if not args.push_to_hub:
        return
    if not args.hub_model_id:
        raise ValueError("--push-to-hub requires --hub-model-id.")

    token_arg: str | bool | None = args.hf_token if args.hf_token else True
    api = HfApi(token=args.hf_token)
    api.create_repo(repo_id=args.hub_model_id, exist_ok=True, token=token_arg)
    api.upload_folder(
        repo_id=args.hub_model_id,
        folder_path=str(folder_path),
        commit_message=f"Upload {description}",
        ignore_patterns=ignore_patterns,
        token=token_arg,
    )
    print(f"{description} pushed to: {args.hub_model_id}")


def copy_best_adapter_checkpoint(args: argparse.Namespace, checkpoint_dir: Path, base_model_path: str) -> Path:
    adapter_output_dir = get_default_adapter_output_dir(args)
    if adapter_output_dir.resolve() != checkpoint_dir.resolve():
        if adapter_output_dir.exists():
            shutil.rmtree(adapter_output_dir)
        shutil.copytree(checkpoint_dir, adapter_output_dir)

    tokenizer = load_tokenizer(args, base_model_path, add_eos_token=False)
    tokenizer.save_pretrained(str(adapter_output_dir))

    metadata = {
        "base_model_id_or_path": base_model_path,
        "checkpoint_path": str(checkpoint_dir),
        "resume_from_checkpoint": str(adapter_output_dir),
        "contains_trainer_state": True,
        "contains_optimizer_state": (adapter_output_dir / "optimizer.pt").exists(),
        "contains_scheduler_state": (adapter_output_dir / "scheduler.pt").exists(),
        "save_merged_model": args.save_merged_model,
    }
    (adapter_output_dir / "adapter_experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(f"Best adapter checkpoint saved to: {adapter_output_dir}")
    return adapter_output_dir


def save_merged_model_and_tokenizer(
    args: argparse.Namespace,
    merged_model,
    tokenizer,
    tokenizer_source_path: str | None = None,
) -> Path:
    merged_output_dir = get_default_merged_output_dir(args)
    merged_output_dir.mkdir(parents=True, exist_ok=True)
    clean_model_output_dir(merged_output_dir)

    if tokenizer_source_path is not None:
        tokenizer = load_tokenizer(args, tokenizer_source_path, add_eos_token=False)
    elif hasattr(tokenizer, "add_eos_token"):
        tokenizer.add_eos_token = False

    set_pad_token(merged_model, tokenizer)
    merged_model.save_pretrained(
        str(merged_output_dir),
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    tokenizer.save_pretrained(str(merged_output_dir))
    print(f"Merged model saved to: {merged_output_dir}")

    return merged_output_dir


def load_tokenizer(args: argparse.Namespace, model_path: str, add_eos_token: bool | None = None):
    tokenizer_kwargs: dict[str, Any] = {
        "cache_dir": str(args.cache_dir),
        "trust_remote_code": args.trust_remote_code,
    }
    if add_eos_token is not None:
        tokenizer_kwargs["add_eos_token"] = add_eos_token

    tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_kwargs)
    tokenizer.padding_side = "right"
    return tokenizer


def load_base_model_for_merge(args: argparse.Namespace, model_path: str):
    compute_dtype = dtype_from_name(args.bnb_4bit_compute_dtype)
    model_kwargs: dict[str, Any] = {
        "device_map": args.device_map,
        "torch_dtype": compute_dtype,
        "cache_dir": str(args.cache_dir),
        "trust_remote_code": args.trust_remote_code,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    return model


def merge_adapter_checkpoint_and_save(args: argparse.Namespace, base_model_path: str, adapter_dir: Path) -> Path:
    print(f"Loading best adapter checkpoint for merge: {adapter_dir}")
    tokenizer = load_tokenizer(args, base_model_path, add_eos_token=False)
    base_model = load_base_model_for_merge(args, base_model_path)
    set_pad_token(base_model, tokenizer)
    peft_model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    merged_model = peft_model.merge_and_unload()
    merged_output_dir = save_merged_model_and_tokenizer(
        args,
        merged_model,
        tokenizer,
        tokenizer_source_path=base_model_path,
    )

    del base_model, peft_model, merged_model
    gc.collect()
    torch.cuda.empty_cache()
    return merged_output_dir


def save_final_adapter_without_best_selection(args: argparse.Namespace, trainer: SFTTrainer, model_path: str) -> Path:
    adapter_output_dir = get_default_adapter_output_dir(args)
    if adapter_output_dir.exists():
        shutil.rmtree(adapter_output_dir)
    trainer.save_model(str(adapter_output_dir))
    tokenizer = load_tokenizer(args, model_path, add_eos_token=False)
    tokenizer.save_pretrained(str(adapter_output_dir))
    metadata = {
        "base_model_id_or_path": model_path,
        "resume_from_checkpoint": None,
        "contains_trainer_state": False,
        "note": "Use output_dir/checkpoint-* to resume training with optimizer state.",
    }
    (adapter_output_dir / "adapter_experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(f"Final adapter saved to: {adapter_output_dir}")
    return adapter_output_dir


def train_and_save(args: argparse.Namespace) -> tuple[Path, str, Path | None]:
    ensure_cuda()
    resolve_precision(args)
    set_seed(args.seed)
    maybe_login(args.hf_token)

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    model_path = download_model_repo(args)
    train_dataset = load_training_dataset(args)
    model, tokenizer = load_model_and_tokenizer(args, model_path)
    model = apply_lora(args, model)

    training_args = make_sft_config(args)
    print_versions()
    trainer = make_trainer(model, tokenizer, train_dataset, training_args)
    print_training_state(trainer)

    output_dir = Path(training_args.output_dir)
    existing_checkpoints = set(list_checkpoint_dirs(output_dir))
    trainer.train(
        resume_from_checkpoint=str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None
    )

    if args.select_best_model:
        checkpoints = [path for path in list_checkpoint_dirs(output_dir) if path not in existing_checkpoints]
        if not checkpoints:
            checkpoints = list_checkpoint_dirs(output_dir)
        if not checkpoints:
            raise RuntimeError("No epoch checkpoints were saved, so best-model selection cannot run.")

        del trainer, model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        best_checkpoint, _ = select_best_epoch_checkpoint(args, model_path, checkpoints)
        adapter_output_dir = copy_best_adapter_checkpoint(args, best_checkpoint, model_path)
        if args.save_merged_model:
            merged_output_dir = merge_adapter_checkpoint_and_save(args, model_path, best_checkpoint)
            return merged_output_dir, model_path, None
        return adapter_output_dir, model_path, adapter_output_dir

    if args.save_merged_model:
        merged_model = trainer.model.merge_and_unload()
        final_output_dir = save_merged_model_and_tokenizer(
            args,
            merged_model,
            tokenizer,
            tokenizer_source_path=model_path,
        )
        del merged_model
        final_peft_path = None
    else:
        final_output_dir = save_final_adapter_without_best_selection(args, trainer, model_path)
        final_peft_path = final_output_dir

    del trainer, model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return final_output_dir, model_path, final_peft_path


def resolve_eval_model(args: argparse.Namespace, merged_output_dir: Path | None) -> str:
    if args.eval_model_id:
        return args.eval_model_id
    if merged_output_dir:
        return str(merged_output_dir)
    if args.merged_output_dir and looks_like_model_dir(args.merged_output_dir):
        return str(args.merged_output_dir)
    raise ValueError(
        "--eval-only/--push-only requires either --eval-peft-path/--adapter-output-dir for adapter mode "
        "or --eval-model-id/--merged-output-dir for full-model mode."
    )


def resolve_eval_peft_path(args: argparse.Namespace, final_peft_path: Path | None) -> Path | None:
    if args.eval_peft_path:
        peft_path = args.eval_peft_path
    elif final_peft_path:
        peft_path = final_peft_path
    elif (args.eval_only or args.push_only) and (
        args.adapter_output_dir or (not args.eval_model_id and not args.merged_output_dir)
    ):
        candidate = get_default_adapter_output_dir(args)
        peft_path = candidate if (candidate / "adapter_config.json").exists() else None
    else:
        peft_path = None

    if peft_path is None:
        return None
    if not peft_path.exists():
        raise FileNotFoundError(f"Adapter path does not exist: {peft_path}")
    if not (peft_path / "adapter_config.json").exists():
        raise FileNotFoundError(f"Adapter config was not found under: {peft_path}")
    return peft_path


def resolve_adapter_eval_base_model(args: argparse.Namespace, base_model_path: str | None) -> str:
    if args.eval_model_id:
        return args.eval_model_id
    if base_model_path:
        return base_model_path

    cached_model_path = resolve_cached_base_model_path(args)
    if cached_model_path:
        return str(cached_model_path)

    expected_local_dir = get_base_local_dir(args)
    print(
        "No local base model repository was found for adapter evaluation. "
        f"Expected {expected_local_dir}. Falling back to model id: {args.model_id}"
    )
    return args.model_id


def resolve_baseline_eval_model(args: argparse.Namespace, base_model_path: str | None) -> str:
    if args.baseline_eval_model_id:
        return args.baseline_eval_model_id
    if base_model_path:
        return base_model_path

    cached_model_path = resolve_cached_base_model_path(args)
    if cached_model_path:
        return str(cached_model_path)

    expected_local_dir = get_base_local_dir(args)
    raise ValueError(
        "Baseline evaluation was requested, but no local base model was found. "
        f"Expected either {expected_local_dir} or a Hugging Face snapshot cache under {args.cache_dir}. "
        "Pass --baseline-eval-model-id with the exact local model directory if it lives elsewhere."
    )


def get_eval_output_path(args: argparse.Namespace, label: str) -> Path:
    if args.eval_output_path is None:
        output_dir = args.cache_dir / "results" / args.eval_tasks
        if args.eval_log_samples:
            return output_dir / label

        filename = f"result-{args.num_fewshot}shot.json"
        if label != "finetuned":
            filename = f"{label}-{filename}"
        return output_dir / filename

    return args.eval_output_path


def make_lm_eval_model_args(
    args: argparse.Namespace,
    eval_model: str,
    peft_path: Path | None = None,
) -> str:
    model_args = [
        f"pretrained={eval_model}",
        f"trust_remote_code={args.trust_remote_code}",
    ]
    if peft_path is not None:
        model_args.append(f"peft={peft_path}")
    if args.eval_dtype.lower() not in {"none", "null", "off"}:
        model_args.append(f"dtype={args.eval_dtype}")
    return ",".join(model_args)


def run_single_eval(
    args: argparse.Namespace,
    label: str,
    eval_model: str,
    output_path: Path,
    peft_path: Path | None = None,
) -> None:
    if args.eval_log_samples and output_path.suffix == "":
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    model_args = make_lm_eval_model_args(args, eval_model, peft_path=peft_path)
    lm_eval_bin = shutil.which("lm_eval")
    if lm_eval_bin:
        cmd = [lm_eval_bin]
    else:
        cmd = [sys.executable, "-m", "lm_eval"]

    cmd.extend(
        [
            "--model",
            "hf",
            "--model_args",
            model_args,
            "--tasks",
            args.eval_tasks,
            "--device",
            args.eval_device,
            "--batch_size",
            str(args.eval_batch_size),
            "--num_fewshot",
            str(args.num_fewshot),
            "--output_path",
            str(output_path),
        ]
    )
    if args.eval_log_samples:
        cmd.append("--log_samples")

    print(f"Running {label} evaluation:")
    print(shlex.join(cmd))
    subprocess.run(cmd, check=True)


def run_lm_eval_for_checkpoint(
    args: argparse.Namespace,
    label: str,
    base_model_path: str,
    adapter_checkpoint: Path,
    output_path: Path,
) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    batch_size = args.best_eval_batch_size or args.eval_batch_size
    model_args = make_lm_eval_model_args(args, base_model_path, peft_path=adapter_checkpoint)
    lm_eval_bin = shutil.which("lm_eval")
    cmd = [lm_eval_bin] if lm_eval_bin else [sys.executable, "-m", "lm_eval"]
    cmd.extend(
        [
            "--model",
            "hf",
            "--model_args",
            model_args,
            "--tasks",
            args.best_eval_tasks,
            "--device",
            args.eval_device,
            "--batch_size",
            str(batch_size),
            "--num_fewshot",
            "0",
            "--output_path",
            str(output_path),
        ]
    )

    print(f"Running zero-shot best-model evaluation for {label}:")
    print(shlex.join(cmd))
    subprocess.run(cmd, check=True)


def get_checkpoint_eval_root(args: argparse.Namespace) -> Path:
    return args.eval_checkpoint_dir or get_default_output_dir(args)


def get_checkpoint_eval_output_root(args: argparse.Namespace) -> Path:
    if args.checkpoint_eval_output_dir:
        return args.checkpoint_eval_output_dir
    return get_default_output_dir(args) / f"checkpoint_{args.num_fewshot}shot_eval" / f"run-{int(time.time())}"


def resolve_checkpoint_eval_targets(args: argparse.Namespace) -> tuple[list[Path], str]:
    if args.eval_checkpoint_paths:
        checkpoints = args.eval_checkpoint_paths
        missing = [path for path in checkpoints if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Some checkpoint paths do not exist:\n" + "\n".join(str(path) for path in missing)
            )

        invalid = [path for path in checkpoints if not (path / "adapter_config.json").exists()]
        if invalid:
            raise FileNotFoundError(
                "Some checkpoint paths do not contain adapter_config.json:\n"
                + "\n".join(str(path) for path in invalid)
            )
        return checkpoints, "selected checkpoint paths"

    checkpoint_root = get_checkpoint_eval_root(args)
    if not checkpoint_root.exists():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_root}")

    all_checkpoints = list_checkpoint_dirs(checkpoint_root)
    checkpoints = [path for path in all_checkpoints if (path / "adapter_config.json").exists()]
    skipped = [path for path in all_checkpoints if path not in checkpoints]
    if skipped:
        print("Skipping checkpoints without adapter_config.json:")
        for checkpoint in skipped:
            print(f"- {checkpoint}")
    if not checkpoints:
        raise RuntimeError(f"No adapter checkpoints found under: {checkpoint_root}")
    return checkpoints, str(checkpoint_root)


def run_checkpoint_evals(args: argparse.Namespace) -> None:
    if args.push_to_hub:
        raise ValueError("--eval-checkpoints cannot be combined with --push-to-hub.")

    checkpoints, checkpoint_source = resolve_checkpoint_eval_targets(args)
    base_model_path = resolve_adapter_eval_base_model(args, base_model_path=None)
    eval_root = get_checkpoint_eval_output_root(args)
    eval_root.mkdir(parents=True, exist_ok=True)

    task = args.eval_tasks.split(",")[0].strip()
    records: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        if args.eval_log_samples:
            output_path = eval_root / checkpoint.name
        else:
            output_path = eval_root / checkpoint.name / f"result-{args.num_fewshot}shot.json"
        run_single_eval(
            args,
            f"{checkpoint.name}_{args.num_fewshot}shot",
            base_model_path,
            output_path,
            peft_path=checkpoint,
        )
        result, result_file = load_lm_eval_result(output_path)
        metric_value = extract_lm_eval_metric(result, task, args.checkpoint_eval_metric)
        epoch = get_checkpoint_epoch(checkpoint)
        record = {
            "checkpoint_path": str(checkpoint),
            "epoch": epoch,
            "task": task,
            "metric": args.checkpoint_eval_metric,
            "metric_value": metric_value,
            "num_fewshot": args.num_fewshot,
            "result_file": str(result_file),
        }
        records.append(record)
        print(f"{checkpoint.name}: {args.checkpoint_eval_metric}={metric_value:.6f}")

    best_record = max(records, key=lambda item: item["metric_value"])
    summary = {
        "checkpoint_source": checkpoint_source,
        "base_model_path": base_model_path,
        "best_checkpoint_path": best_record["checkpoint_path"],
        "best_metric_value": best_record["metric_value"],
        "selection_metric": args.checkpoint_eval_metric,
        "eval_tasks": args.eval_tasks,
        "num_fewshot": args.num_fewshot,
        "records": records,
    }
    summary_path = eval_root / "checkpoint_eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Checkpoint evaluation summary saved to: {summary_path}")
    print(
        "Best checkpoint: "
        f"{best_record['checkpoint_path']} ({args.checkpoint_eval_metric}={best_record['metric_value']:.6f})"
    )


def load_lm_eval_result(output_path: Path) -> tuple[dict[str, Any], Path]:
    candidates = [output_path] if output_path.is_file() else sorted(
        output_path.rglob("*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        if "sample" in candidate.name.lower():
            continue
        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(data, dict) and "results" in data:
            return data, candidate

    raise FileNotFoundError(f"Could not find an lm_eval result JSON under {output_path}.")


def extract_lm_eval_metric(result: dict[str, Any], task: str, metric: str) -> float:
    results = result.get("results", {})
    task_result = results.get(task)
    if task_result is None and results:
        first_task = next(iter(results))
        print(f"Task {task!r} not found in lm_eval results. Falling back to {first_task!r}.")
        task_result = results[first_task]

    if not isinstance(task_result, dict):
        raise KeyError(f"No task metrics found for {task!r}.")

    if metric in task_result:
        return float(task_result[metric])

    for key, value in task_result.items():
        if key == metric or key.startswith(f"{metric},"):
            return float(value)

    raise KeyError(f"Metric {metric!r} not found for task {task!r}. Available metrics: {sorted(task_result)}")


def select_best_epoch_checkpoint(
    args: argparse.Namespace,
    base_model_path: str,
    checkpoints: list[Path],
) -> tuple[Path, dict[str, Any]]:
    eval_root = args.best_eval_output_dir
    if eval_root is None:
        eval_root = get_default_output_dir(args) / "epoch_zero_shot_eval" / f"run-{int(time.time())}"
    eval_root.mkdir(parents=True, exist_ok=True)

    task = args.best_eval_tasks.split(",")[0].strip()
    records: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        output_path = eval_root / checkpoint.name
        run_lm_eval_for_checkpoint(args, checkpoint.name, base_model_path, checkpoint, output_path)
        result, result_file = load_lm_eval_result(output_path)
        metric_value = extract_lm_eval_metric(result, task, args.best_eval_metric)
        epoch = get_checkpoint_epoch(checkpoint)
        record = {
            "checkpoint_path": str(checkpoint),
            "epoch": epoch,
            "task": task,
            "metric": args.best_eval_metric,
            "metric_value": metric_value,
            "result_file": str(result_file),
        }
        records.append(record)
        print(f"{checkpoint.name}: {args.best_eval_metric}={metric_value:.6f}")

    best_record = max(records, key=lambda item: item["metric_value"])
    summary = {
        "best_checkpoint_path": best_record["checkpoint_path"],
        "best_metric_value": best_record["metric_value"],
        "selection_metric": args.best_eval_metric,
        "selection_tasks": args.best_eval_tasks,
        "num_fewshot": 0,
        "records": records,
    }
    summary_path = eval_root / "best_epoch_eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Best checkpoint summary saved to: {summary_path}")
    print(
        "Best checkpoint: "
        f"{best_record['checkpoint_path']} ({args.best_eval_metric}={best_record['metric_value']:.6f})"
    )
    return Path(best_record["checkpoint_path"]), best_record


def run_eval(
    args: argparse.Namespace,
    final_output_dir: Path | None,
    base_model_path: str | None = None,
    final_peft_path: Path | None = None,
) -> dict[str, Any]:
    if args.eval_baseline:
        label = "baseline"
        eval_model = resolve_baseline_eval_model(args, base_model_path)
        peft_path = None
    else:
        peft_path = resolve_eval_peft_path(args, final_peft_path)
        if peft_path is not None:
            label = "finetuned_adapter"
            eval_model = resolve_adapter_eval_base_model(args, base_model_path)
        else:
            label = "finetuned"
            eval_model = resolve_eval_model(args, final_output_dir)

    output_path = get_eval_output_path(args, label)
    run_single_eval(args, label, eval_model, output_path, peft_path=peft_path)
    return {
        "label": label,
        "eval_model": eval_model,
        "output_path": output_path,
        "peft_path": peft_path,
    }


def handle_finetuned_artifacts(args: argparse.Namespace, eval_info: dict[str, Any]) -> None:
    if not args.save_merged_model and not args.push_to_hub:
        return
    if not (args.eval_only or args.push_only):
        print("Artifact push/save is only active with --eval-only or --push-only. Skipping for this training run.")
        return
    if args.eval_baseline:
        if args.push_to_hub:
            raise ValueError("--push-to-hub is not supported with --eval-baseline.")
        return

    peft_path = eval_info["peft_path"]
    eval_model = eval_info["eval_model"]
    merged_output_dir = None
    if peft_path is not None:
        if args.save_merged_model:
            ensure_cuda()
            resolve_precision(args)
            merged_output_dir = merge_adapter_checkpoint_and_save(args, str(eval_model), peft_path)
        if not args.push_to_hub:
            return
        if merged_output_dir is not None:
            push_folder_to_hub(args, merged_output_dir, "selected merged model")
        else:
            push_folder_to_hub(
                args,
                peft_path,
                "selected adapter",
                ignore_patterns=HUB_ADAPTER_IGNORE_PATTERNS,
            )
        return

    if not args.push_to_hub:
        return

    eval_model_path = Path(str(eval_model)).expanduser()
    if not eval_model_path.is_dir():
        raise ValueError(
            "--push-to-hub without adapter evaluation requires a local model directory from "
            "--eval-model-id or --merged-output-dir."
        )
    push_folder_to_hub(args, eval_model_path, "selected model")


def resolve_push_only_artifact(args: argparse.Namespace) -> dict[str, Any]:
    if not args.push_to_hub:
        raise ValueError("--push-only requires --push-to-hub.")
    if args.eval_baseline:
        raise ValueError("--push-only cannot be used with --eval-baseline.")

    peft_path = resolve_eval_peft_path(args, final_peft_path=None)
    if peft_path is not None:
        return {
            "label": "finetuned_adapter",
            "eval_model": resolve_adapter_eval_base_model(args, base_model_path=None),
            "output_path": None,
            "peft_path": peft_path,
        }

    return {
        "label": "finetuned",
        "eval_model": resolve_eval_model(args, merged_output_dir=None),
        "output_path": None,
        "peft_path": None,
    }


def main() -> None:
    args = parse_args()
    if args.push_only:
        maybe_login(args.hf_token)
        artifact_info = resolve_push_only_artifact(args)
        handle_finetuned_artifacts(args, artifact_info)
        return

    if args.eval_only:
        maybe_login(args.hf_token)
        if args.eval_checkpoints:
            run_checkpoint_evals(args)
            return
        eval_info = run_eval(args, final_output_dir=None)
        handle_finetuned_artifacts(args, eval_info)
        return

    if args.push_to_hub:
        print("--push-to-hub is only active with --eval-only or --push-only. This training run will save locally only.")

    final_output_dir, base_model_path, final_peft_path = train_and_save(args)
    if args.run_eval:
        run_eval(
            args,
            final_output_dir,
            base_model_path=base_model_path,
            final_peft_path=final_peft_path,
        )


if __name__ == "__main__":
    main()
