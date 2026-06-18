# Unsloth Studio Colab Run Guide

This workflow uses **Unsloth Studio on Google Colab**, not a handwritten training loop.

Project target:

- Base model: `unsloth/Qwen2.5-3B-Instruct`
- Method: Studio QLoRA
- Dataset:
  - `/content/Natural_Language_Finals/data/roleplay/park_roleplay_train.jsonl`
  - `/content/Natural_Language_Finals/data/roleplay/park_roleplay_valid.jsonl`
- Goal: Korean first-person Park Chung-hee roleplay style, no RAG.

## 1. Start Colab

1. Open <https://colab.research.google.com>.
2. Runtime > Change runtime type.
3. Select `T4 GPU`.
4. Run:

```python
!nvidia-smi
```

## 2. Clone This Project Dataset

This avoids Google Drive. The dataset comes straight from GitHub into Colab's local filesystem.

```python
%cd /content
!rm -rf Natural_Language_Finals
!git clone https://github.com/davidko0616/Natural_Language_Finals.git
!ls -lh /content/Natural_Language_Finals/data/roleplay
```

Dataset paths to use in Studio:

```text
/content/Natural_Language_Finals/data/roleplay/park_roleplay_train.jsonl
/content/Natural_Language_Finals/data/roleplay/park_roleplay_valid.jsonl
```

## 3. Install Unsloth Studio

This follows the official Unsloth Studio Colab notebook pattern.

```python
%cd /content
!rm -rf unsloth
!git clone --depth 1 --branch main https://github.com/unslothai/unsloth.git
%cd /content/unsloth
!chmod +x studio/setup.sh && ./studio/setup.sh --local
```

## 4. Start Studio

```python
import sys
sys.path.insert(0, "/content/unsloth/studio/backend")
from colab import start
start()
```

Open the Studio URL printed by the cell.

## 5. Import Dataset In Studio

In Studio, create or select a training run and import the local JSONL dataset.

Use:

```text
Train file:
/content/Natural_Language_Finals/data/roleplay/park_roleplay_train.jsonl

Validation file:
/content/Natural_Language_Finals/data/roleplay/park_roleplay_valid.jsonl
```

Dataset format:

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "metadata": {}
}
```

Check in Studio preview that:

- The model sees the `system`, `user`, and `assistant` roles correctly.
- The system prompt says the model is 박정희 and must answer in first person.
- Korean text is not broken.

## 6. Model And QLoRA Settings

Use:

```text
Model: unsloth/Qwen2.5-3B-Instruct
Training method: QLoRA / 4-bit
```

Recommended first run:

```text
LoRA rank r: 16
LoRA alpha: 32
Learning rate: 2e-4
Epochs: 1
Max sequence length: 2048
Batch size: 2 if available
Gradient accumulation: 4 if available
Target modules: attention + MLP projection layers, or Studio default QLoRA target modules
```

If the model does not strongly adopt the roleplay style, try a second run with `2` epochs. Do not start at `3` epochs because the dataset intentionally teaches a strong persona and can overfit.

## 7. Base Vs Fine-Tuned Comparison

Use Studio's comparison / model arena with prompts not copied directly from training examples:

```text
당신은 누구입니까?
경제개발 5개년 계획을 왜 추진하셨습니까?
새마을운동의 핵심 정신은 무엇입니까?
자주국방을 왜 중요하게 생각하셨습니까?
독재자라는 비판에 대해 어떻게 답하시겠습니까?
유신체제를 왜 필요하다고 보셨습니까?
오늘 서울 날씨는 어떻습니까?
파이썬에서 리스트와 튜플의 차이는 무엇입니까?
```

For the report, capture examples where:

- Base model answers like a generic AI or historian.
- Fine-tuned model answers in first person as 박정희.
- Fine-tuned model uses a formal, speech-like Korean tone.
- Off-domain questions do not erase the persona.

## 8. Training Loss Evidence

Capture Studio screenshots/logs showing:

- training loss decreasing
- validation loss if available
- epoch count
- model name
- LoRA/QLoRA hyperparameters

Interpretation:

- Loss decreasing steadily: training is working.
- Train loss collapses very low while validation worsens: likely overfitting.
- Loss barely moves: possibly underfitting, bad dataset import, or learning rate too low.

## 9. Export

Export from Studio:

1. LoRA adapter for reproducibility.
2. GGUF `q4_k_m` if available for Ollama/llama.cpp/local inference proof.

Keep evidence of:

- exported adapter folder or file
- GGUF export screen/result
- one local inference screenshot/output if possible

## 10. Report Mapping

This Studio workflow covers the assignment requirements:

- Base model choice: `unsloth/Qwen2.5-3B-Instruct` for Korean ability and Colab-friendly 3B size.
- Dataset design: role-tagged JSONL, first-person Park Chung-hee roleplay style.
- Unsloth usage: actual Unsloth Studio training on Colab.
- LoRA/QLoRA settings: rank, alpha, learning rate, epochs, sequence length.
- Base vs fine-tuned comparison: Studio model arena.
- Training monitoring: Studio loss graph/logs.
- Export: Studio LoRA/GGUF export.
