"""
Colab QLoRA fine-tuning script for the Park Chung-hee roleplay model.

Run on Google Colab with a CUDA GPU:

    !pip install -U unsloth
    !python training/colab_unsloth_qwen25_park_roleplay.py --epochs 1

Expected dataset files:
    data/roleplay/park_roleplay_train.jsonl
    data/roleplay/park_roleplay_valid.jsonl

Each JSONL row must contain:
    {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}
"""

from __future__ import annotations

import argparse
import gc
import json
import urllib.request
from inspect import signature
from pathlib import Path


DEFAULT_MODEL = "unsloth/Qwen2.5-3B-Instruct"
DEFAULT_REPO_RAW_BASE = "https://raw.githubusercontent.com/davidko0616/Natural_Language_Finals/main"
DEFAULT_TRAIN_FILE = "data/roleplay/park_roleplay_train.jsonl"
DEFAULT_VALID_FILE = "data/roleplay/park_roleplay_valid.jsonl"
DEFAULT_OUTPUT_DIR = "outputs/unsloth_qwen25_park"
DEFAULT_ADAPTER_DIR = "outputs/unsloth_qwen25_park/adapter"
DEFAULT_GGUF_DIR = "outputs/unsloth_qwen25_park/gguf_q4_k_m"

EVAL_PROMPTS = [
    "당신은 누구입니까?",
    "경제개발 5개년 계획을 왜 추진하셨습니까?",
    "새마을운동의 핵심 정신은 무엇입니까?",
    "자주국방을 왜 중요하게 생각하셨습니까?",
    "독재자라는 비판에 대해 어떻게 답하시겠습니까?",
    "유신체제를 왜 필요하다고 보셨습니까?",
    "오늘 서울 날씨는 어떻습니까?",
    "파이썬에서 리스트와 튜플의 차이는 무엇입니까?",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default=DEFAULT_MODEL)
    parser.add_argument("--train_file", default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--valid_file", default=DEFAULT_VALID_FILE)
    parser.add_argument("--repo_raw_base", default=DEFAULT_REPO_RAW_BASE)
    parser.add_argument("--no_download_dataset", action="store_true")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--adapter_dir", default=DEFAULT_ADAPTER_DIR)
    parser.add_argument("--gguf_dir", default=DEFAULT_GGUF_DIR)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--max_new_tokens", type=int, default=320)
    parser.add_argument("--export_gguf", action="store_true")
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def download_file(url: str, path: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] {url} -> {target}")
    urllib.request.urlretrieve(url, target)


def ensure_dataset_files(args: argparse.Namespace) -> None:
    missing = [p for p in [args.train_file, args.valid_file] if not Path(p).exists()]
    if missing and args.no_download_dataset:
        raise FileNotFoundError(
            "Missing dataset file(s): "
            + ", ".join(missing)
            + "\nDataset download is disabled by --no_download_dataset."
        )
    for path in missing:
        normalized_path = path.replace("\\", "/")
        url = f"{args.repo_raw_base.rstrip('/')}/{normalized_path}"
        download_file(url, path)
    still_missing = [p for p in [args.train_file, args.valid_file] if not Path(p).exists()]
    if still_missing:
        raise FileNotFoundError(
            "Missing dataset file(s): "
            + ", ".join(still_missing)
            + "\nCheck --repo_raw_base or upload the files manually."
        )


def validate_jsonl(path: str) -> int:
    count = 0
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            messages = row.get("messages")
            roles = [m.get("role") for m in messages or []]
            if roles != ["system", "user", "assistant"]:
                raise ValueError(f"{path}:{line_no} has invalid roles: {roles}")
            count += 1
    return count


def build_text_dataset(dataset, tokenizer):
    def format_batch(examples):
        texts = [
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            for messages in examples["messages"]
        ]
        return {"text": texts}

    return dataset.map(format_batch, batched=True, remove_columns=dataset.column_names)


def make_sft_config(args, is_bfloat16_supported):
    from trl import SFTConfig

    common = dict(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_ratio=0.05,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        logging_steps=10,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        seed=args.seed,
        report_to="none",
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=False,
    )
    try:
        return SFTConfig(eval_strategy="steps", save_strategy="steps", **common)
    except TypeError:
        return SFTConfig(evaluation_strategy="steps", save_strategy="steps", **common)


def build_trainer(model, tokenizer, train_dataset, valid_dataset, training_args, max_seq_length):
    from trl import SFTTrainer

    kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
    )
    params = signature(SFTTrainer.__init__).parameters
    if "processing_class" in params:
        kwargs["processing_class"] = tokenizer
    else:
        kwargs["tokenizer"] = tokenizer
    if "dataset_text_field" in params:
        kwargs["dataset_text_field"] = "text"
    if "max_seq_length" in params:
        kwargs["max_seq_length"] = max_seq_length
    if "packing" in params:
        kwargs["packing"] = False
    return SFTTrainer(**kwargs)


def maybe_train_on_responses_only(trainer):
    try:
        from unsloth.chat_templates import train_on_responses_only

        return train_on_responses_only(
            trainer,
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n",
        )
    except Exception as exc:
        print(f"[warn] train_on_responses_only was skipped: {exc}")
        return trainer


def generate_responses(model, tokenizer, prompts, system_prompt, max_new_tokens):
    import torch

    outputs = []
    for prompt in prompts:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.08,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = generated[:, inputs.input_ids.shape[1] :]
        response = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
        outputs.append({"prompt": prompt, "response": response})
    return outputs


def load_system_prompt(train_file: str) -> str:
    with Path(train_file).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                return row["messages"][0]["content"]
    raise ValueError(f"No rows found in {train_file}")


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    ensure_dataset_files(args)
    train_count = validate_jsonl(args.train_file)
    valid_count = validate_jsonl(args.valid_file)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print(f"[data] train rows: {train_count}")
    print(f"[data] valid rows: {valid_count}")
    print(f"[model] {args.model_name}")

    # Unsloth must be imported before transformers/trl model setup.
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from datasets import load_dataset

    max_seq_length = args.max_seq_length
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw = load_dataset(
        "json",
        data_files={"train": args.train_file, "validation": args.valid_file},
    )
    train_dataset = build_text_dataset(raw["train"], tokenizer)
    valid_dataset = build_text_dataset(raw["validation"], tokenizer)
    system_prompt = load_system_prompt(args.train_file)

    print("[eval] generating base model comparison outputs")
    model.eval()
    base_outputs = generate_responses(
        model,
        tokenizer,
        EVAL_PROMPTS,
        system_prompt,
        max_new_tokens=args.max_new_tokens,
    )
    save_json(Path(args.output_dir) / "base_outputs.json", base_outputs)

    print("[train] attaching LoRA adapters")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        use_rslora=False,
        loftq_config=None,
    )

    training_args = make_sft_config(args, is_bfloat16_supported)
    trainer = build_trainer(
        model,
        tokenizer,
        train_dataset,
        valid_dataset,
        training_args,
        max_seq_length,
    )
    trainer = maybe_train_on_responses_only(trainer)

    print("[train] starting QLoRA training")
    train_result = trainer.train()
    eval_result = trainer.evaluate()

    print("[save] saving LoRA adapter")
    model.save_pretrained(args.adapter_dir)
    tokenizer.save_pretrained(args.adapter_dir)

    history = trainer.state.log_history
    save_json(Path(args.output_dir) / "training_log_history.json", history)
    save_json(
        Path(args.output_dir) / "training_summary.json",
        {
            "model_name": args.model_name,
            "train_file": args.train_file,
            "valid_file": args.valid_file,
            "train_rows": train_count,
            "valid_rows": valid_count,
            "max_seq_length": max_seq_length,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.grad_accum,
            "train_metrics": train_result.metrics,
            "eval_metrics": eval_result,
        },
    )

    print("[eval] generating fine-tuned model comparison outputs")
    FastLanguageModel.for_inference(model)
    finetuned_outputs = generate_responses(
        model,
        tokenizer,
        EVAL_PROMPTS,
        system_prompt,
        max_new_tokens=args.max_new_tokens,
    )
    comparison = [
        {
            "prompt": prompt,
            "base": base["response"],
            "finetuned": tuned["response"],
        }
        for prompt, base, tuned in zip(EVAL_PROMPTS, base_outputs, finetuned_outputs)
    ]
    save_json(Path(args.output_dir) / "finetuned_outputs.json", finetuned_outputs)
    save_json(Path(args.output_dir) / "base_vs_finetuned.json", comparison)

    print("\n===== BASE VS FINE-TUNED SAMPLE =====")
    for row in comparison[:4]:
        print(f"\n[Prompt] {row['prompt']}")
        print(f"[Base]\n{row['base']}")
        print(f"[Fine-tuned]\n{row['finetuned']}")

    if args.export_gguf:
        print("[export] attempting GGUF q4_k_m export")
        try:
            model.save_pretrained_gguf(
                args.gguf_dir,
                tokenizer,
                quantization_method="q4_k_m",
            )
        except Exception as exc:
            save_json(Path(args.output_dir) / "gguf_export_error.json", {"error": str(exc)})
            print(f"[warn] GGUF export failed: {exc}")

    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass
    print(f"[done] outputs saved under {args.output_dir}")


if __name__ == "__main__":
    main()
