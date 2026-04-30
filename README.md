# Mistral-7B ARC QLoRA Experiments

This repo contains two main scripts:

- `run_mistral_arc_experiment.py`: QLoRA training, evaluation, adapter save/push.
- `analyze_choice_similarity.py`: cosine similarity analysis among MCQA answer choices.

The default base model is `mistralai/Mistral-7B-v0.1`. Training saves LoRA
adapters by default. A merged full model is saved only when explicitly requested.

## Setup

Use an Ubuntu server with a CUDA-capable NVIDIA GPU. A100 is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
CUDA_VERSION=cu121 bash install.sh
```

`install.sh` installs PyTorch first with the selected CUDA wheel, then installs
`requirements.txt`.

If you want FlashAttention:

```bash
INSTALL_FLASH_ATTN=1 CUDA_VERSION=cu121 bash install.sh
```

For Hugging Face private/gated access or push:

```bash
export HF_TOKEN=hf_your_token
# or
huggingface-cli login
```

## Training Datasets

Choose the training data with `--train-dataset`.

- `alpaca`: 500 sampled Alpaca examples by default.
- `arc`: ARC-Easy + ARC-Challenge train splits.
- `sciqa`: SciQ/SciQA train split.
- `openbookqa`: OpenBookQA train split.
- `alpaca_arc`: Alpaca + ARC.
- `sciqa_openbookqa`: SciQA + OpenBookQA.
- `arc_sciqa_openbookqa`: ARC + SciQA + OpenBookQA.
- `all`: Alpaca + ARC + SciQA + OpenBookQA.

Useful dataset size options:

```bash
--sample-size 500
--arc-sample-size 0
--sciqa-sample-size 0
--openbookqa-sample-size 0
```

`0` means use all examples for that MCQA dataset.

Prompt format for MCQA datasets:

```bash
--arc-format question_answer
--arc-format question_choices_answer
```

`question_answer` uses only the question before `Answer:`.
`question_choices_answer` also shows all answer choices before `Answer:`.

## Baseline SFT Loss

Standard SFT trains the model to generate the correct answer text. This is the
default loss.

```bash
python run_mistral_arc_experiment.py \
  --cache-dir /tmp/huggingface_cache \
  --output-dir /tmp/huggingface_cache/runs/arc_sft \
  --adapter-output-dir /tmp/huggingface_cache/adapters/arc_sft_best \
  --train-dataset arc \
  --arc-format question_answer \
  --loss-type sft
```

By default, the script saves epoch checkpoints, runs zero-shot `lm_eval`, and
copies the checkpoint with the best `acc_norm` to `--adapter-output-dir`.

Use a unique `--output-dir` and `--adapter-output-dir` for every experiment to
avoid checkpoint collisions.

## Contrastive Choice Loss

The contrastive loss compares all answer choices for the same question:

```text
-log softmax(log P(choice | question) / tau)[correct_choice]
```

Use it only with MCQA datasets: `arc`, `sciqa`, `openbookqa`,
`sciqa_openbookqa`, or `arc_sciqa_openbookqa`. Do not use it with Alpaca.

```bash
python run_mistral_arc_experiment.py \
  --cache-dir /tmp/huggingface_cache \
  --output-dir /tmp/huggingface_cache/runs/arc_contrastive \
  --adapter-output-dir /tmp/huggingface_cache/adapters/arc_contrastive_best \
  --train-dataset arc \
  --arc-format question_answer \
  --loss-type choice_contrastive \
  --choice-loss-temperature 1.0 \
  --choice-loss-length-normalize
```

`--choice-loss-length-normalize` uses average token log-probability for each
choice, which is closer to `lm_eval`'s `acc_norm`.

## Evaluation

Evaluate the saved best adapter:

```bash
python run_mistral_arc_experiment.py \
  --cache-dir /tmp/huggingface_cache \
  --eval-only \
  --eval-model-id /tmp/huggingface_cache/mistralai/Mistral-7B-v0.1 \
  --eval-peft-path /tmp/huggingface_cache/adapters/arc_contrastive_best \
  --num-fewshot 25
```

If `lm_eval` fails because of a `dtype` argument mismatch:

```bash
--eval-dtype none
```

Evaluate selected checkpoints:

```bash
python run_mistral_arc_experiment.py \
  --cache-dir /tmp/huggingface_cache \
  --eval-only \
  --eval-checkpoints \
  --eval-checkpoint-paths \
    /tmp/huggingface_cache/runs/arc_contrastive/checkpoint-120 \
    /tmp/huggingface_cache/runs/arc_contrastive/checkpoint-240 \
  --num-fewshot 25
```

## Push To Hugging Face

Push a known-good adapter without running eval:

```bash
HF_TOKEN=hf_your_token python run_mistral_arc_experiment.py \
  --cache-dir /tmp/huggingface_cache \
  --push-only \
  --adapter-output-dir /tmp/huggingface_cache/adapters/arc_contrastive_best \
  --hub-model-id your-hf-id/mistral-7b-arc-contrastive-adapter \
  --push-to-hub
```

Merge the adapter into the base model and push a full model:

```bash
HF_TOKEN=hf_your_token python run_mistral_arc_experiment.py \
  --cache-dir /tmp/huggingface_cache \
  --push-only \
  --adapter-output-dir /tmp/huggingface_cache/adapters/arc_contrastive_best \
  --save-merged-model \
  --merged-output-dir /tmp/huggingface_cache/merged/arc_contrastive \
  --hub-model-id your-hf-id/mistral-7b-arc-contrastive-merged \
  --push-to-hub
```

## Choice Similarity Analysis

Analyze cosine similarity among answer choices for ARC, SciQA, and OpenBookQA:

```bash
python analyze_choice_similarity.py \
  --cache-dir /tmp/huggingface_cache \
  --output-dir /tmp/huggingface_cache/choice_similarity \
  --datasets arc sciqa openbookqa \
  --splits train test
```

Quick smoke test:

```bash
python analyze_choice_similarity.py \
  --cache-dir /tmp/huggingface_cache \
  --output-dir /tmp/huggingface_cache/choice_similarity_smoke \
  --sample-size 200
```

Outputs:

- `choice_similarity_records.csv`
- `choice_similarity_summary.json`
- `dataset_comparison.png`
- `arc_train_test_similarity.png`
- `sciqa_train_test_similarity.png`
- `openbookqa_train_test_similarity.png`
