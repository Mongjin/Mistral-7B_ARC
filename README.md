# Mistral-7B ARC QLoRA Experiment

This repository contains a server-friendly Python script that reproduces the
notebook experiment in `Mistal-7B_ARC.ipynb`.

Default experiment:

- Base model: `mistralai/Mistral-7B-v0.1`
- Dataset: `tatsu-lab/alpaca`, `train` split by default
- Sampling: 500 examples after `shuffle(seed=42)`
- Optional MCQA training data: ARC, SciQA, and OpenBookQA train splits
- Fine-tuning: QLoRA, `r=8`, `lora_alpha=16`, `target_modules=["q_proj", "v_proj"]`
- Training: 1 epoch, batch size 4, learning rate `2e-4`, max length 1024
- Precision: auto bf16 on supported GPUs such as A100, otherwise fp16
- Best model selection: after training, evaluate each epoch checkpoint with zero-shot `arc_challenge` and save the adapter checkpoint with the highest `acc_norm`
- Final save: adapter-only by default; use `--save-merged-model` only when a full merged model is needed
- Evaluation: `lm-evaluation-harness`, `arc_challenge`, 25-shot, batch size 8

## Ubuntu setup

The commands below assume an Ubuntu server with an NVIDIA driver, an A100 GPU,
Python 3.10 or newer, and a working CUDA-compatible environment.

```bash
python3 -m venv .venv
source .venv/bin/activate

# Use cu121 by default. Override CUDA_VERSION if your server uses another
# PyTorch CUDA wheel channel, for example cu124.
CUDA_VERSION=cu121 bash install.sh
```

FlashAttention is optional. If you want to enable it on A100:

```bash
INSTALL_FLASH_ATTN=1 CUDA_VERSION=cu121 bash install.sh
```

If Hugging Face access is required, log in before running:

```bash
huggingface-cli login
```

Alternatively, set `HF_TOKEN` in the environment.

## Run training and ARC evaluation

```bash
source .venv/bin/activate

python run_mistral_arc_experiment.py \
  --cache-dir /tmp/huggingface_cache \
  --adapter-output-dir /tmp/huggingface_cache/mistral-7b-qlora-alpaca-sample-0.5k-adapter
```

The script trains the LoRA adapter, saves the best adapter checkpoint, clears
GPU memory, and then runs ARC Challenge evaluation with `pretrained=base,peft=adapter`.

By default, the script saves a LoRA checkpoint at every epoch, runs zero-shot
`lm_eval` on each epoch checkpoint, selects the checkpoint with the highest
`acc_norm`, and copies that best checkpoint to the final adapter directory. This
keeps the adapter plus Trainer state needed for resume. The epoch-selection
summary is written under:

```bash
/tmp/huggingface_cache/results/epoch_zero_shot_eval/
```

Useful best-model selection options:

- `--no-select-best-model` disables epoch-wise zero-shot selection and saves the final epoch model directly.
- `--best-eval-tasks arc_challenge` controls the zero-shot task used for checkpoint selection.
- `--best-eval-metric acc_norm` controls the metric used for selection.
- `--best-eval-batch-size 8` overrides the batch size for epoch-wise zero-shot evaluation.
- `--best-eval-output-dir /path/to/dir` changes where epoch-wise evaluation JSON files are written.
- `--resume-from-checkpoint /path/to/checkpoint` resumes training from a saved checkpoint with optimizer/scheduler state.
- `--save-merged-model` additionally saves a full merged model locally. Hub push is handled from `--eval-only`.

To train on ARC Easy and ARC Challenge train examples instead of Alpaca:

```bash
python run_mistral_arc_experiment.py \
  --cache-dir /tmp/huggingface_cache \
  --train-dataset arc \
  --adapter-output-dir /tmp/huggingface_cache/mistral-7b-qlora-arc-train-adapter
```

To train ARC with every answer choice shown before `Answer:`:

```bash
python run_mistral_arc_experiment.py \
  --cache-dir /tmp/huggingface_cache \
  --train-dataset arc \
  --arc-format question_choices_answer \
  --adapter-output-dir /tmp/huggingface_cache/mistral-7b-qlora-arc-choices-adapter
```

To mix the original 500 Alpaca samples with all ARC train examples:

```bash
python run_mistral_arc_experiment.py \
  --cache-dir /tmp/huggingface_cache \
  --train-dataset alpaca_arc \
  --adapter-output-dir /tmp/huggingface_cache/mistral-7b-qlora-alpaca-arc-adapter
```

To train on SciQA:

```bash
python run_mistral_arc_experiment.py \
  --cache-dir /tmp/huggingface_cache \
  --train-dataset sciqa \
  --adapter-output-dir /tmp/huggingface_cache/mistral-7b-qlora-sciqa-adapter
```

To train on OpenBookQA:

```bash
python run_mistral_arc_experiment.py \
  --cache-dir /tmp/huggingface_cache \
  --train-dataset openbookqa \
  --adapter-output-dir /tmp/huggingface_cache/mistral-7b-qlora-openbookqa-adapter
```

To train on both SciQA and OpenBookQA:

```bash
python run_mistral_arc_experiment.py \
  --cache-dir /tmp/huggingface_cache \
  --train-dataset sciqa_openbookqa \
  --adapter-output-dir /tmp/huggingface_cache/mistral-7b-qlora-sciqa-openbookqa-adapter
```

MCQA dataset options:

- `--arc-configs ARC-Easy ARC-Challenge` controls which ARC subsets are loaded.
- `--arc-split train` controls the ARC split.
- `--arc-sample-size 0` means use all combined ARC examples. Set a positive number to sample after combining and shuffling.
- `--sciqa-dataset-id allenai/sciq` controls the SciQA dataset id. The old `--sciq-dataset-id` spelling is still accepted.
- `--sciqa-split train` controls the SciQA split. The old `--sciq-split` spelling is still accepted.
- `--sciqa-sample-size 0` means use all SciQA examples. The old `--sciq-sample-size` spelling is still accepted.
- `--openbookqa-dataset-id allenai/openbookqa` controls the OpenBookQA dataset id.
- `--openbookqa-configs main` controls which OpenBookQA configs are loaded. Use `main additional` to combine both.
- `--openbookqa-split train` controls the OpenBookQA split.
- `--openbookqa-sample-size 0` means use all combined OpenBookQA examples. Set a positive number to sample after combining and shuffling.
- `--arc-format question_answer` keeps the original format: `{question} Answer: {answer_text}`.
- `--arc-format question_choices_answer` uses `{question}\nChoices:\nA. ...\nB. ...\nAnswer: {answer_text}` for ARC, SciQA, and OpenBookQA.
- `--train-dataset sciq`, `sicq`, `sicqa`, and `science_qa` are accepted as deprecated aliases for `sciqa`.
- `--train-dataset sciqa_openbookqa` combines SciQA and OpenBookQA.
- `--train-dataset arc_sciqa_openbookqa` combines ARC, SciQA, and OpenBookQA.
- `--train-dataset all` combines Alpaca, ARC, SciQA, and OpenBookQA.

To evaluate the original Mistral-7B baseline instead of the fine-tuned model:

```bash
python run_mistral_arc_experiment.py \
  --cache-dir /tmp/huggingface_cache \
  --eval-baseline
```

Each script run executes exactly one evaluation. Without `--eval-baseline`, the
script evaluates the fine-tuned model. With `--eval-baseline`, it evaluates the
base model and writes `baseline-result-25shot.json` by default.

To use FlashAttention after installing it:

```bash
python run_mistral_arc_experiment.py \
  --cache-dir /tmp/huggingface_cache \
  --attn-implementation flash_attention_2
```

To train only and skip evaluation:

```bash
python run_mistral_arc_experiment.py --no-run-eval
```

To evaluate an already saved local model or Hugging Face model:

```bash
python run_mistral_arc_experiment.py \
  --eval-only \
  --eval-model-id /tmp/huggingface_cache/mistral-7b-qlora-alpaca-sample-0.5k
```

To evaluate an adapter-only result, `lm_eval` must load both the original
Mistral backbone and the LoRA adapter. If the backbone is already under
`/tmp/huggingface_cache/mistralai/Mistral-7B-v0.1`, this is enough:

```bash
python run_mistral_arc_experiment.py \
  --cache-dir /tmp/huggingface_cache \
  --eval-only \
  --adapter-output-dir /tmp/huggingface_cache/mistral-7b-qlora-arc-train-adapter
```

Use `--eval-model-id /path/to/base-mistral` when the backbone lives somewhere
else, and use `--eval-peft-path /path/to/adapter` when you do not want to use
`--adapter-output-dir`.

To evaluate only the unfine-tuned baseline:

```bash
python run_mistral_arc_experiment.py \
  --cache-dir /tmp/huggingface_cache \
  --eval-only \
  --eval-baseline
```

For baseline evaluation, the script only uses a local base model by default. It
first checks `/tmp/huggingface_cache/mistralai/Mistral-7B-v0.1`, then existing
Hugging Face snapshot cache directories under `/tmp/huggingface_cache`. Use
`--base-local-dir` or `--baseline-eval-model-id` when the local model lives
somewhere else.

If `lm_eval` fails with an error such as
`MistralForCausalLM.__init__() got an unexpected keyword argument 'dtype'`,
omit the evaluation dtype argument:

```bash
python run_mistral_arc_experiment.py \
  --eval-only \
  --eval-model-id /tmp/huggingface_cache/mistralai/Mistral-7B-v0.1 \
  --eval-peft-path /tmp/huggingface_cache/mistral-7b-qlora-arc-train-adapter \
  --eval-dtype none
```

For error analysis, save per-sample inputs, predictions, references, and
metrics:

```bash
python run_mistral_arc_experiment.py \
  --eval-only \
  --eval-model-id /tmp/huggingface_cache/mistralai/Mistral-7B-v0.1 \
  --eval-peft-path /tmp/huggingface_cache/mistral-7b-qlora-arc-train-adapter \
  --eval-dtype none \
  --eval-log-samples
```

With sample logging enabled, the default output directory is
`/tmp/huggingface_cache/results/arc_challenge/finetuned_adapter/` for adapter
evaluation. For baseline evaluation it is
`/tmp/huggingface_cache/results/arc_challenge/baseline/`.
Inside that directory, `lm_eval` writes the aggregate result plus a task-level
sample file that can be filtered for correct and incorrect ARC examples.

To evaluate only selected saved checkpoints with 25-shot ARC Challenge:

```bash
python run_mistral_arc_experiment.py \
  --cache-dir /tmp/huggingface_cache \
  --eval-only \
  --eval-checkpoints \
  --eval-checkpoint-paths \
    /tmp/huggingface_cache/results/checkpoint-120 \
    /tmp/huggingface_cache/results/checkpoint-240 \
  --num-fewshot 25
```

Each selected checkpoint directory is evaluated as
`pretrained=base,peft=checkpoint`. By default, result JSON files and
`checkpoint_eval_summary.json` are written under:

```bash
/tmp/huggingface_cache/results/checkpoint_25shot_eval/
```

Use `--checkpoint-eval-output-dir /path/to/eval-results` to choose a different
output directory.

If you intentionally want to evaluate every checkpoint under a directory, omit
`--eval-checkpoint-paths` and use:

```bash
python run_mistral_arc_experiment.py \
  --cache-dir /tmp/huggingface_cache \
  --eval-only \
  --eval-checkpoints \
  --eval-checkpoint-dir /tmp/huggingface_cache/results \
  --num-fewshot 25
```

## Optional Hub push

Pushing is disabled by default to avoid accidental uploads. Training runs save
locally only. To decide based on evaluation results, run `--eval-only` first;
adding `--push-to-hub` uploads the evaluated artifact after `lm_eval` succeeds.
For adapter evaluation, the upload contains adapter/tokenizer files and skips
local resume state such as optimizer/scheduler/RNG files:

```bash
HF_TOKEN=your_token_here python run_mistral_arc_experiment.py \
  --cache-dir /tmp/huggingface_cache \
  --eval-only \
  --adapter-output-dir /tmp/huggingface_cache/mistral-7b-qlora-arc-train-adapter \
  --hub-model-id your-hf-id/mistral-7b-qlora-arc-train-adapter \
  --push-to-hub
```

If you already know which fine-tuned adapter is best and do not want to run
`lm_eval`, use `--push-only`:

```bash
HF_TOKEN=your_token_here python run_mistral_arc_experiment.py \
  --cache-dir /tmp/huggingface_cache \
  --push-only \
  --adapter-output-dir /tmp/huggingface_cache/mistral-7b-qlora-arc-train-adapter \
  --hub-model-id your-hf-id/mistral-7b-qlora-arc-train-adapter \
  --push-to-hub
```

To evaluate an adapter, save a full merged model locally, and push that merged
model instead:

```bash
HF_TOKEN=your_token_here python run_mistral_arc_experiment.py \
  --cache-dir /tmp/huggingface_cache \
  --eval-only \
  --adapter-output-dir /tmp/huggingface_cache/mistral-7b-qlora-arc-train-adapter \
  --save-merged-model \
  --merged-output-dir /tmp/huggingface_cache/mistral-7b-qlora-arc-train-merged \
  --hub-model-id your-hf-id/mistral-7b-qlora-arc-train-merged \
  --push-to-hub
```

Omit `--push-to-hub` from the command above if you only want to evaluate and
save the merged model locally.

To skip evaluation and directly merge/push a known-good adapter:

```bash
HF_TOKEN=your_token_here python run_mistral_arc_experiment.py \
  --cache-dir /tmp/huggingface_cache \
  --push-only \
  --adapter-output-dir /tmp/huggingface_cache/mistral-7b-qlora-arc-train-adapter \
  --save-merged-model \
  --merged-output-dir /tmp/huggingface_cache/mistral-7b-qlora-arc-train-merged \
  --hub-model-id your-hf-id/mistral-7b-qlora-arc-train-merged \
  --push-to-hub
```

## Useful A100 options

The script follows the notebook's auto precision behavior. On A100 it uses bf16
by default. To force a specific mode:

```bash
python run_mistral_arc_experiment.py --precision bf16
python run_mistral_arc_experiment.py --precision fp16
```
