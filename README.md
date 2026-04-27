# Mistral-7B ARC QLoRA Experiment

This repository contains a server-friendly Python script that reproduces the
notebook experiment in `Mistal-7B_ARC.ipynb`.

Default experiment:

- Base model: `mistralai/Mistral-7B-v0.1`
- Dataset: `tatsu-lab/alpaca`, `train` split
- Sampling: 500 examples after `shuffle(seed=42)`
- Fine-tuning: QLoRA, `r=8`, `lora_alpha=16`, `target_modules=["q_proj", "v_proj"]`
- Training: 1 epoch, batch size 4, learning rate `2e-4`, max length 1024
- Precision: auto bf16 on supported GPUs such as A100, otherwise fp16
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
  --merged-output-dir /tmp/huggingface_cache/mistral-7b-qlora-alpaca-sample-0.5k
```

The script trains the LoRA adapter, merges it into the base model, saves the
merged model, clears GPU memory, and then runs ARC Challenge evaluation.

To evaluate the original Mistral-7B baseline instead of the fine-tuned model:

```bash
python run_mistral_arc_experiment.py \
  --cache-dir /tmp/huggingface_cache \
  --merged-output-dir /tmp/huggingface_cache/mistral-7b-qlora-alpaca-sample-0.5k \
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
  --eval-model-id /tmp/huggingface_cache/mistral-7b-qlora-alpaca-sample-0.5k \
  --eval-dtype none
```

## Optional Hub push

Pushing is disabled by default to avoid accidental uploads. To push the merged
model and tokenizer:

```bash
HF_TOKEN=your_token_here python run_mistral_arc_experiment.py \
  --hub-model-id your-hf-id/mistral-7b-qlora-alpaca-sample-0.5k \
  --push-to-hub
```

## Useful A100 options

The script follows the notebook's auto precision behavior. On A100 it uses bf16
by default. To force a specific mode:

```bash
python run_mistral_arc_experiment.py --precision bf16
python run_mistral_arc_experiment.py --precision fp16
```
