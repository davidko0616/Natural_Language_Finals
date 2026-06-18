# Colab Unsloth Run Guide

This project trains a no-RAG Park Chung-hee roleplay model with:

- Base model: `unsloth/Qwen2.5-3B-Instruct`
- Method: QLoRA with Unsloth
- Dataset:
  - `data/roleplay/park_roleplay_train.jsonl`
  - `data/roleplay/park_roleplay_valid.jsonl`

## 1. Start Colab

1. Open <https://colab.research.google.com>.
2. Runtime > Change runtime type.
3. Select `T4 GPU`.
4. Run:

```python
!nvidia-smi
```

## 2. Install Unsloth

```python
!pip install -U unsloth
```

If Colab asks you to restart the runtime after installation, restart and rerun the install cell once.

## 3. Get The Training Script Into Colab

Fast path, without Google Drive and without cloning the whole repo:

```python
!mkdir -p training
!wget -O training/colab_unsloth_qwen25_park_roleplay.py \
  https://raw.githubusercontent.com/davidko0616/Natural_Language_Finals/main/training/colab_unsloth_qwen25_park_roleplay.py
```

The script will automatically download these dataset files from the same GitHub repository if they are missing:

- `data/roleplay/park_roleplay_train.jsonl`
- `data/roleplay/park_roleplay_valid.jsonl`

Alternative: clone the whole repository.

```python
!git clone https://github.com/davidko0616/Natural_Language_Finals.git
%cd Natural_Language_Finals
```

## 4. Verify Dataset Files

If you used the fast path, run training once or explicitly call the script; it will create `data/roleplay/` and download the files.

```python
!python training/colab_unsloth_qwen25_park_roleplay.py --help
```

Expected files:

- `park_roleplay_train.jsonl`
- `park_roleplay_valid.jsonl`
- optionally `park_roleplay_all.jsonl`
- optionally `dataset_report.json`

## 5. Train With QLoRA

Recommended first run:

```python
!python training/colab_unsloth_qwen25_park_roleplay.py --epochs 1
```

For a fork or different branch, point the script at a different raw GitHub base:

```python
!python training/colab_unsloth_qwen25_park_roleplay.py \
  --repo_raw_base https://raw.githubusercontent.com/USER/REPO/BRANCH \
  --epochs 1
```

If the validation loss is still improving and output is not yet stylistically strong enough, run a second experiment:

```python
!python training/colab_unsloth_qwen25_park_roleplay.py \
  --epochs 2 \
  --output_dir outputs/unsloth_qwen25_park_e2 \
  --adapter_dir outputs/unsloth_qwen25_park_e2/adapter
```

Do not start with 3 epochs. The dataset is persona-heavy, so overfitting/memorization is possible.

## 6. Export GGUF

Try GGUF export only after the adapter training succeeds:

```python
!python training/colab_unsloth_qwen25_park_roleplay.py \
  --epochs 1 \
  --output_dir outputs/unsloth_qwen25_park_gguf \
  --adapter_dir outputs/unsloth_qwen25_park_gguf/adapter \
  --export_gguf
```

This can take extra time and disk space. If it fails, keep the LoRA adapter and save the error file as evidence.

## 7. Files To Download For The Report

After training, download:

- `outputs/unsloth_qwen25_park/training_summary.json`
- `outputs/unsloth_qwen25_park/training_log_history.json`
- `outputs/unsloth_qwen25_park/base_outputs.json`
- `outputs/unsloth_qwen25_park/finetuned_outputs.json`
- `outputs/unsloth_qwen25_park/base_vs_finetuned.json`
- `outputs/unsloth_qwen25_park/adapter/`
- `outputs/unsloth_qwen25_park/gguf_q4_k_m/` if GGUF export succeeds

These directly support the required sections:

- model choice
- dataset design
- Unsloth training proof
- LoRA/QLoRA hyperparameters
- training loss monitoring
- base vs fine-tuned comparison
- export/local execution evidence

## 8. Hyperparameter Rationale

Initial settings:

- `r=16`: enough capacity for style/persona adaptation without training too many parameters.
- `alpha=32`: common 2x rank scaling for stronger LoRA updates.
- `learning_rate=2e-4`: standard QLoRA starting point for SFT.
- `epochs=1`: safer first pass because this dataset intentionally teaches a strong persona.
- `batch_size=2`, `gradient_accumulation=4`: effective batch size 8 while fitting T4 memory.
- `max_seq_length=2048`: enough for the generated answers while keeping T4 memory stable.

## 9. Evaluation Prompts

The script automatically compares base vs fine-tuned outputs on:

- `당신은 누구입니까?`
- `경제개발 5개년 계획을 왜 추진하셨습니까?`
- `새마을운동의 핵심 정신은 무엇입니까?`
- `자주국방을 왜 중요하게 생각하셨습니까?`
- `독재자라는 비판에 대해 어떻게 답하시겠습니까?`
- `유신체제를 왜 필요하다고 보셨습니까?`
- `오늘 서울 날씨는 어떻습니까?`
- `파이썬에서 리스트와 튜플의 차이는 무엇입니까?`

For the final report, use examples where the base model answers like a generic assistant but the fine-tuned model answers in first person as 박정희.
