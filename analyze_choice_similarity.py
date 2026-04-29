from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer


DEFAULT_ARC_DATASET_ID = "allenai/ai2_arc"
DEFAULT_SCIQA_DATASET_ID = "allenai/sciq"
DEFAULT_OPENBOOKQA_DATASET_ID = "allenai/openbookqa"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class ChoiceExample:
    dataset: str
    split: str
    subset: str
    example_id: str
    choices: list[str]


@dataclass
class SimilarityRecord:
    dataset: str
    split: str
    subset: str
    example_id: str
    num_choices: int
    pair_count: int
    mean_cosine: float
    min_cosine: float
    max_cosine: float
    std_cosine: float
    pairwise_cosines: list[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze pairwise cosine similarity among answer choices for ARC, "
            "SciQA/SciQ, and OpenBookQA train/test splits."
        )
    )
    parser.add_argument("--datasets", nargs="+", default=["arc", "sciqa", "openbookqa"])
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/huggingface_cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("choice_similarity_results"))
    parser.add_argument("--sample-size", type=int, default=0, help="Per dataset/split sample size. <=0 uses all rows.")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--arc-dataset-id", default=DEFAULT_ARC_DATASET_ID)
    parser.add_argument("--arc-configs", nargs="+", default=["ARC-Easy", "ARC-Challenge"])
    parser.add_argument("--sciqa-dataset-id", "--sciq-dataset-id", dest="sciqa_dataset_id", default=DEFAULT_SCIQA_DATASET_ID)
    parser.add_argument("--openbookqa-dataset-id", default=DEFAULT_OPENBOOKQA_DATASET_ID)
    parser.add_argument("--openbookqa-configs", nargs="+", default=["main"])

    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding-batch-size", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-pairwise-csv", action="store_true")

    args = parser.parse_args()
    args.cache_dir = args.cache_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.datasets = [normalize_dataset_name(name) for name in args.datasets]
    return args


def normalize_dataset_name(name: str) -> str:
    aliases = {
        "sciq": "sciqa",
        "sicq": "sciqa",
        "sicqa": "sciqa",
        "scienceqa": "sciqa",
        "science_qa": "sciqa",
        "open_book_qa": "openbookqa",
        "openbook": "openbookqa",
    }
    normalized = aliases.get(name.lower(), name.lower())
    if normalized not in {"arc", "sciqa", "openbookqa"}:
        raise ValueError(f"Unsupported dataset: {name}. Use arc, sciqa, or openbookqa.")
    return normalized


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def resolve_dtype(dtype: str, device: torch.device) -> torch.dtype:
    if dtype == "float32" or device.type == "cpu":
        return torch.float32
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if device.type == "cuda":
        return torch.float16
    return torch.float32


def load_embedding_model(args: argparse.Namespace):
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.embedding_model,
        cache_dir=str(args.cache_dir),
        trust_remote_code=args.trust_remote_code,
    )
    model = AutoModel.from_pretrained(
        args.embedding_model,
        cache_dir=str(args.cache_dir),
        trust_remote_code=args.trust_remote_code,
        torch_dtype=dtype,
    )
    model.to(device)
    model.eval()
    print(f"Embedding model: {args.embedding_model}")
    print(f"Embedding device: {device}, dtype: {dtype}")
    return tokenizer, model, device


def sample_dataset(dataset, sample_size: int, seed: int):
    if sample_size <= 0 or sample_size >= len(dataset):
        return dataset
    return dataset.shuffle(seed=seed).select(range(sample_size))


def sample_examples(examples: list[ChoiceExample], sample_size: int, seed: int) -> list[ChoiceExample]:
    if sample_size <= 0 or sample_size >= len(examples):
        return examples
    rng = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(examples), generator=rng)[:sample_size].tolist()
    return [examples[index] for index in indices]


def safe_get_id(example: dict[str, Any], fallback_idx: int) -> str:
    for key in ("id", "question_id", "uid"):
        value = example.get(key)
        if value is not None:
            return str(value)
    return str(fallback_idx)


def normalize_choice_text(choice: Any) -> str:
    return re.sub(r"\s+", " ", str(choice)).strip()


def get_arc_examples(args: argparse.Namespace, split: str) -> list[ChoiceExample]:
    examples: list[ChoiceExample] = []
    for config in args.arc_configs:
        dataset = load_dataset(args.arc_dataset_id, config, split=split, cache_dir=str(args.cache_dir))
        for idx, row in enumerate(dataset):
            choices = [normalize_choice_text(text) for text in row["choices"]["text"]]
            choices = [choice for choice in choices if choice]
            examples.append(ChoiceExample("arc", split, config, safe_get_id(row, idx), choices))
        print(f"Loaded {len(dataset)} rows from {args.arc_dataset_id}/{config}/{split}.")
    return sample_examples(examples, args.sample_size, args.seed)


def get_sciqa_examples(args: argparse.Namespace, split: str) -> list[ChoiceExample]:
    dataset = load_dataset(args.sciqa_dataset_id, split=split, cache_dir=str(args.cache_dir))
    dataset = sample_dataset(dataset, args.sample_size, args.seed)
    examples: list[ChoiceExample] = []
    for idx, row in enumerate(dataset):
        choices = [
            normalize_choice_text(row["correct_answer"]),
            normalize_choice_text(row["distractor1"]),
            normalize_choice_text(row["distractor2"]),
            normalize_choice_text(row["distractor3"]),
        ]
        choices = [choice for choice in choices if choice]
        examples.append(ChoiceExample("sciqa", split, "main", safe_get_id(row, idx), choices))
    print(f"Loaded {len(dataset)} rows from {args.sciqa_dataset_id}/{split}.")
    return examples


def get_openbookqa_examples(args: argparse.Namespace, split: str) -> list[ChoiceExample]:
    examples: list[ChoiceExample] = []
    for config in args.openbookqa_configs:
        dataset = load_dataset(args.openbookqa_dataset_id, config, split=split, cache_dir=str(args.cache_dir))
        for idx, row in enumerate(dataset):
            choices = [normalize_choice_text(text) for text in row["choices"]["text"]]
            choices = [choice for choice in choices if choice]
            examples.append(ChoiceExample("openbookqa", split, config, safe_get_id(row, idx), choices))
        print(f"Loaded {len(dataset)} rows from {args.openbookqa_dataset_id}/{config}/{split}.")
    return sample_examples(examples, args.sample_size, args.seed)


def load_choice_examples(args: argparse.Namespace) -> list[ChoiceExample]:
    loaders = {
        "arc": get_arc_examples,
        "sciqa": get_sciqa_examples,
        "openbookqa": get_openbookqa_examples,
    }
    examples: list[ChoiceExample] = []
    for dataset_name in args.datasets:
        for split in args.splits:
            examples.extend(loaders[dataset_name](args, split))
    examples = [example for example in examples if len(example.choices) >= 2]
    if not examples:
        raise RuntimeError("No examples with at least two answer choices were loaded.")
    print(f"Loaded {len(examples)} total examples with at least two choices.")
    return examples


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(dtype=last_hidden_state.dtype)
    pooled = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)
    return pooled / counts


@torch.no_grad()
def encode_texts(
    texts: list[str],
    tokenizer,
    model,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> dict[str, torch.Tensor]:
    embeddings: dict[str, torch.Tensor] = {}
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        batch = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        outputs = model(**batch)
        pooled = mean_pool(outputs.last_hidden_state, batch["attention_mask"])
        pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=1).cpu()
        for text, embedding in zip(batch_texts, pooled):
            embeddings[text] = embedding
        print(f"Encoded {min(start + batch_size, len(texts))}/{len(texts)} unique choices.", end="\r")
    print()
    return embeddings


def compute_similarity_records(
    examples: list[ChoiceExample],
    embeddings: dict[str, torch.Tensor],
) -> list[SimilarityRecord]:
    records: list[SimilarityRecord] = []
    for example in examples:
        choice_embeddings = torch.stack([embeddings[choice] for choice in example.choices])
        similarity_matrix = choice_embeddings @ choice_embeddings.T
        pairwise: list[float] = []
        for i in range(len(example.choices)):
            for j in range(i + 1, len(example.choices)):
                pairwise.append(float(similarity_matrix[i, j].item()))
        if not pairwise:
            continue
        mean_value = sum(pairwise) / len(pairwise)
        variance = sum((value - mean_value) ** 2 for value in pairwise) / len(pairwise)
        records.append(
            SimilarityRecord(
                dataset=example.dataset,
                split=example.split,
                subset=example.subset,
                example_id=example.example_id,
                num_choices=len(example.choices),
                pair_count=len(pairwise),
                mean_cosine=mean_value,
                min_cosine=min(pairwise),
                max_cosine=max(pairwise),
                std_cosine=math.sqrt(variance),
                pairwise_cosines=pairwise,
            )
        )
    return records


def quantile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    sorted_values = sorted(values)
    position = (len(sorted_values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def summarize_records(records: list[SimilarityRecord]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[SimilarityRecord]] = {}
    for record in records:
        groups.setdefault((record.dataset, record.split), []).append(record)

    summary: list[dict[str, Any]] = []
    for (dataset_name, split), group_records in sorted(groups.items()):
        means = [record.mean_cosine for record in group_records]
        pairwise = [value for record in group_records for value in record.pairwise_cosines]
        mean_of_means = sum(means) / len(means)
        pairwise_mean = sum(pairwise) / len(pairwise)
        summary.append(
            {
                "dataset": dataset_name,
                "split": split,
                "num_examples": len(group_records),
                "num_pairs": len(pairwise),
                "mean_example_mean_cosine": mean_of_means,
                "median_example_mean_cosine": quantile(means, 0.5),
                "q25_example_mean_cosine": quantile(means, 0.25),
                "q75_example_mean_cosine": quantile(means, 0.75),
                "pairwise_mean_cosine": pairwise_mean,
                "pairwise_median_cosine": quantile(pairwise, 0.5),
            }
        )
    return summary


def write_outputs(args: argparse.Namespace, records: list[SimilarityRecord], summary: list[dict[str, Any]]) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)

    record_path = args.output_dir / "choice_similarity_records.csv"
    with record_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "split",
                "subset",
                "example_id",
                "num_choices",
                "pair_count",
                "mean_cosine",
                "min_cosine",
                "max_cosine",
                "std_cosine",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "dataset": record.dataset,
                    "split": record.split,
                    "subset": record.subset,
                    "example_id": record.example_id,
                    "num_choices": record.num_choices,
                    "pair_count": record.pair_count,
                    "mean_cosine": record.mean_cosine,
                    "min_cosine": record.min_cosine,
                    "max_cosine": record.max_cosine,
                    "std_cosine": record.std_cosine,
                }
            )

    summary_path = args.output_dir / "choice_similarity_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.save_pairwise_csv:
        pairwise_path = args.output_dir / "choice_similarity_pairwise.csv"
        with pairwise_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["dataset", "split", "subset", "example_id", "pair_index", "cosine"],
            )
            writer.writeheader()
            for record in records:
                for pair_index, cosine in enumerate(record.pairwise_cosines):
                    writer.writerow(
                        {
                            "dataset": record.dataset,
                            "split": record.split,
                            "subset": record.subset,
                            "example_id": record.example_id,
                            "pair_index": pair_index,
                            "cosine": cosine,
                        }
                    )

    print(f"Records saved to: {record_path}")
    print(f"Summary saved to: {summary_path}")


def records_by_dataset_split(records: list[SimilarityRecord]) -> dict[tuple[str, str], list[SimilarityRecord]]:
    grouped: dict[tuple[str, str], list[SimilarityRecord]] = {}
    for record in records:
        grouped.setdefault((record.dataset, record.split), []).append(record)
    return grouped


def prettify_dataset_name(name: str) -> str:
    return {"arc": "ARC", "sciqa": "SciQA", "openbookqa": "OpenBookQA"}.get(name, name)


def plot_overall_comparison(args: argparse.Namespace, records: list[SimilarityRecord]) -> None:
    grouped = records_by_dataset_split(records)
    datasets = [name for name in args.datasets if any(key[0] == name for key in grouped)]
    splits = [split for split in args.splits if any(key[1] == split for key in grouped)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    box_data: list[list[float]] = []
    box_labels: list[str] = []
    positions: list[float] = []
    colors: list[str] = []
    split_colors = {"train": "#4C78A8", "test": "#F58518", "validation": "#54A24B"}

    position = 1.0
    for dataset_name in datasets:
        for split in splits:
            values = [record.mean_cosine for record in grouped.get((dataset_name, split), [])]
            if not values:
                continue
            box_data.append(values)
            box_labels.append(f"{prettify_dataset_name(dataset_name)}\n{split}")
            positions.append(position)
            colors.append(split_colors.get(split, "#777777"))
            position += 1.0
        position += 0.5

    box = axes[0].boxplot(box_data, positions=positions, patch_artist=True, showfliers=False)
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(box_labels, rotation=30, ha="right")
    axes[0].set_ylabel("Per-question mean pairwise cosine")
    axes[0].set_title("Choice Similarity Distribution")
    axes[0].grid(axis="y", alpha=0.25)

    bar_labels: list[str] = []
    bar_means: list[float] = []
    bar_colors: list[str] = []
    for dataset_name in datasets:
        for split in splits:
            values = [record.mean_cosine for record in grouped.get((dataset_name, split), [])]
            if not values:
                continue
            bar_labels.append(f"{prettify_dataset_name(dataset_name)}\n{split}")
            bar_means.append(sum(values) / len(values))
            bar_colors.append(split_colors.get(split, "#777777"))
    axes[1].bar(range(len(bar_means)), bar_means, color=bar_colors, alpha=0.75)
    axes[1].set_xticks(range(len(bar_labels)))
    axes[1].set_xticklabels(bar_labels, rotation=30, ha="right")
    axes[1].set_ylabel("Mean of per-question mean cosine")
    axes[1].set_title("Dataset/Split Mean Similarity")
    axes[1].grid(axis="y", alpha=0.25)

    output_path = args.output_dir / "dataset_comparison.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Overall comparison plot saved to: {output_path}")


def plot_dataset_train_test(args: argparse.Namespace, records: list[SimilarityRecord]) -> None:
    grouped = records_by_dataset_split(records)
    split_colors = {"train": "#4C78A8", "test": "#F58518", "validation": "#54A24B"}

    for dataset_name in args.datasets:
        dataset_records = [record for record in records if record.dataset == dataset_name]
        if not dataset_records:
            continue
        available_splits = [split for split in args.splits if (dataset_name, split) in grouped]
        if not available_splits:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
        for split in available_splits:
            values = [record.mean_cosine for record in grouped[(dataset_name, split)]]
            axes[0].hist(
                values,
                bins=30,
                density=True,
                alpha=0.45,
                label=f"{split} (n={len(values)})",
                color=split_colors.get(split, None),
            )
        axes[0].set_title(f"{prettify_dataset_name(dataset_name)} Train/Test Distribution")
        axes[0].set_xlabel("Per-question mean pairwise cosine")
        axes[0].set_ylabel("Density")
        axes[0].legend()
        axes[0].grid(axis="y", alpha=0.25)

        box_data = [[record.mean_cosine for record in grouped[(dataset_name, split)]] for split in available_splits]
        box = axes[1].boxplot(box_data, labels=available_splits, patch_artist=True, showfliers=False)
        for patch, split in zip(box["boxes"], available_splits):
            patch.set_facecolor(split_colors.get(split, "#777777"))
            patch.set_alpha(0.65)
        axes[1].set_title(f"{prettify_dataset_name(dataset_name)} Split Comparison")
        axes[1].set_ylabel("Per-question mean pairwise cosine")
        axes[1].grid(axis="y", alpha=0.25)

        output_path = args.output_dir / f"{dataset_name}_train_test_similarity.png"
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
        print(f"Dataset split plot saved to: {output_path}")


def plot_results(args: argparse.Namespace, records: list[SimilarityRecord]) -> None:
    plot_overall_comparison(args, records)
    plot_dataset_train_test(args, records)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    examples = load_choice_examples(args)
    unique_choices = sorted({choice for example in examples for choice in example.choices})
    print(f"Unique choice texts to embed: {len(unique_choices)}")

    tokenizer, model, device = load_embedding_model(args)
    embeddings = encode_texts(
        unique_choices,
        tokenizer,
        model,
        device,
        batch_size=args.embedding_batch_size,
        max_length=args.max_length,
    )
    records = compute_similarity_records(examples, embeddings)
    summary = summarize_records(records)
    write_outputs(args, records, summary)
    plot_results(args, records)


if __name__ == "__main__":
    main()
