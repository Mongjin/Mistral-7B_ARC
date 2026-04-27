from __future__ import annotations

import argparse
import inspect
import os
import shlex
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from datasets import concatenate_datasets, load_dataset
from huggingface_hub import login, snapshot_download
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
from trl import SFTConfig, SFTTrainer


DEFAULT_MODEL_ID = "mistralai/Mistral-7B-v0.1"
DEFAULT_DATASET_ID = "tatsu-lab/alpaca"
DEFAULT_ARC_DATASET_ID = "allenai/ai2_arc"
DEFAULT_MODEL_NAME = "mistral-7b-qlora-alpaca-sample-0.5k"
TRAIN_COLUMNS = ["instruction", "input", "output", "text", "source"]


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
        choices=["alpaca", "arc", "alpaca_arc"],
        default="alpaca",
        help="Training data source. 'arc' uses AI2 ARC-Easy and ARC-Challenge train splits.",
    )
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--arc-dataset-id", default=DEFAULT_ARC_DATASET_ID)
    parser.add_argument("--arc-configs", nargs="+", default=["ARC-Easy", "ARC-Challenge"])
    parser.add_argument("--arc-split", default="train")
    parser.add_argument(
        "--arc-sample-size",
        type=int,
        default=0,
        help="Number of combined ARC examples to use. Use 0 or a negative value for all ARC train examples.",
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("/home/mongjin/tmp/huggingface_cache"))
    parser.add_argument("--base-local-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--merged-output-dir", type=Path, default=None)
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--hub-model-id", default=None)
    parser.add_argument("--push-to-hub", action="store_true")
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
    parser.add_argument("--save-steps", type=int, default=50)
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
    parser.add_argument("--run-eval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--eval-baseline",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Evaluate the base Mistral model instead of the fine-tuned model.",
    )
    parser.add_argument(
        "--baseline-eval-model-id",
        default="mistralai/Mistral-7B-v0.1",
        help="Model id or local path for baseline evaluation. Defaults to the downloaded base model path when available.",
    )
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-model-id", default=None)
    parser.add_argument("--eval-output-path", type=Path, default=None)
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
    args.cache_dir = args.cache_dir.expanduser().resolve()
    if args.base_local_dir is not None:
        args.base_local_dir = args.base_local_dir.expanduser().resolve()
    if args.output_dir is not None:
        args.output_dir = args.output_dir.expanduser().resolve()
    if args.merged_output_dir is not None:
        args.merged_output_dir = args.merged_output_dir.expanduser().resolve()
    if args.eval_output_path is not None:
        args.eval_output_path = args.eval_output_path.expanduser().resolve()
    return args


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
    return args.cache_dir / DEFAULT_MODEL_NAME


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
    text = f"<s>[INST] {instruction} here are the inputs {input_text} [/INST] \n {output} </s>"
    return {
        "instruction": instruction,
        "input": input_text,
        "output": output,
        "text": text,
        "source": "alpaca",
    }


def format_arc_example(example: dict[str, Any]) -> dict[str, str]:
    choices = example["choices"]
    labels = [str(label) for label in choices["label"]]
    texts = [str(text) for text in choices["text"]]
    answer = str(example["answerKey"])
    answer_text = texts[labels.index(answer)] if answer in labels else answer
    prompt = f"Question: {example['question']}\nAnswer:"
    text = f"<s>[INST] {prompt} [/INST]\n{answer_text}</s>"
    return {
        "instruction": "Answer the science question.",
        "input": prompt,
        "output": answer_text,
        "text": text,
        "source": "arc",
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
        dataset = dataset.map(format_arc_example).select_columns(TRAIN_COLUMNS)
        dataset = dataset.remove_columns("source").add_column("source", [config] * len(dataset))
        arc_datasets.append(dataset)
        print(f"Loaded {len(dataset)} examples from {args.arc_dataset_id}/{config}/{args.arc_split}.")

    dataset = concatenate_datasets(arc_datasets)
    dataset = sample_dataset(dataset, args.arc_sample_size, args.seed, "ARC")
    return dataset


def load_training_dataset(args: argparse.Namespace):
    datasets = []
    if args.train_dataset in {"alpaca", "alpaca_arc"}:
        datasets.append(load_alpaca_training_dataset(args))
    if args.train_dataset in {"arc", "alpaca_arc"}:
        datasets.append(load_arc_training_dataset(args))

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


def train_and_save(args: argparse.Namespace) -> tuple[Path, str]:
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

    trainer.train()

    merged_model = trainer.model.merge_and_unload()
    merged_output_dir = get_default_merged_output_dir(args)
    merged_output_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(str(merged_output_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(merged_output_dir))
    print(f"Merged model saved to: {merged_output_dir}")

    if args.push_to_hub:
        if not args.hub_model_id:
            raise ValueError("--push-to-hub requires --hub-model-id.")
        token_arg: str | bool | None = args.hf_token if args.hf_token else True
        merged_model.push_to_hub(
            repo_id=args.hub_model_id,
            token=token_arg,
            max_shard_size=args.max_shard_size,
            safe_serialization=True,
        )
        tokenizer.push_to_hub(repo_id=args.hub_model_id, token=token_arg)
        print(f"Merged model pushed to: {args.hub_model_id}")

    del trainer, model, merged_model
    torch.cuda.empty_cache()
    return merged_output_dir, model_path


def resolve_eval_model(args: argparse.Namespace, merged_output_dir: Path | None) -> str:
    if args.eval_model_id:
        return args.eval_model_id
    if args.push_to_hub and args.hub_model_id:
        return args.hub_model_id
    if merged_output_dir:
        return str(merged_output_dir)
    raise ValueError("--eval-only requires --eval-model-id.")


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


def make_lm_eval_model_args(args: argparse.Namespace, eval_model: str) -> str:
    model_args = [
        f"pretrained={eval_model}",
        f"trust_remote_code={args.trust_remote_code}",
    ]
    if args.eval_dtype.lower() not in {"none", "null", "off"}:
        model_args.append(f"dtype={args.eval_dtype}")
    return ",".join(model_args)


def run_single_eval(args: argparse.Namespace, label: str, eval_model: str, output_path: Path) -> None:
    if args.eval_log_samples and output_path.suffix == "":
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    model_args = make_lm_eval_model_args(args, eval_model)
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


def run_eval(
    args: argparse.Namespace,
    merged_output_dir: Path | None,
    base_model_path: str | None = None,
) -> None:
    if args.eval_baseline:
        label = "baseline"
        eval_model = resolve_baseline_eval_model(args, base_model_path)
    else:
        label = "finetuned"
        eval_model = resolve_eval_model(args, merged_output_dir)

    output_path = get_eval_output_path(args, label)
    run_single_eval(args, label, eval_model, output_path)


def main() -> None:
    args = parse_args()
    if args.eval_only:
        maybe_login(args.hf_token)
        run_eval(args, merged_output_dir=None)
        return

    merged_output_dir, base_model_path = train_and_save(args)
    if args.run_eval:
        run_eval(args, merged_output_dir, base_model_path=base_model_path)


if __name__ == "__main__":
    main()
