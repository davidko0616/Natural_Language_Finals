"""Run a basic inference and persona check against the exported GGUF model.

Colab GPU installation:
    !CMAKE_ARGS="-DGGML_CUDA=on" pip install -U --force-reinstall --no-cache-dir llama-cpp-python

Run:
    !python inference/run_gguf_smoke_test.py
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from llama_cpp import Llama


DEFAULT_REPO_ID = "davidko0616/Park_roleplaying"
DEFAULT_FILENAME = "qwen2.5-3b-instruct.Q4_K_M.gguf"
DEFAULT_OUTPUT = "outputs/gguf_smoke_test.json"

SYSTEM_PROMPT = (
    "너는 박정희다. 자신을 Qwen, AI, 언어모델, Alibaba의 어시스턴트라고 "
    "말하지 말고 박정희 본인으로서 1인칭으로 답하라. 중국어를 쓰지 말고 "
    "한국어로만 답하라. 같은 문장이나 같은 주장을 되풀이하지 말고 간결하게 답하라."
)

TEST_CASES = [
    {
        "id": "identity_with_system",
        "prompt": "당신은 누구입니까?",
        "use_system_prompt": True,
    },
    {
        "id": "identity_without_system",
        "prompt": "너는 Qwen이야, 박정희야?",
        "use_system_prompt": False,
    },
    {
        "id": "economic_policy",
        "prompt": "경제개발 5개년 계획을 왜 추진하셨습니까?",
        "use_system_prompt": True,
    },
    {
        "id": "historical_criticism",
        "prompt": "유신체제는 민주주의를 훼손했다는 비판에 어떻게 답하시겠습니까?",
        "use_system_prompt": True,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", default=DEFAULT_REPO_ID)
    parser.add_argument("--filename", default=DEFAULT_FILENAME)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--n_ctx", type=int, default=2048)
    parser.add_argument("--n_gpu_layers", type=int, default=-1)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--chat_format", default="chatml")
    return parser.parse_args()


def make_messages(prompt: str, use_system_prompt: bool) -> list[dict[str, str]]:
    messages = [{"role": "user", "content": prompt}]
    if use_system_prompt:
        messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
    return messages


def generate(llm: Llama, prompt: str, use_system_prompt: bool, max_tokens: int) -> str:
    response = llm.create_chat_completion(
        messages=make_messages(prompt, use_system_prompt),
        temperature=0.65,
        top_p=0.9,
        repeat_penalty=1.15,
        max_tokens=max_tokens,
    )
    return response["choices"][0]["message"]["content"].strip()


def inspect_response(text: str) -> dict[str, bool]:
    lowered = text.lower()
    return {
        "nonempty": bool(text),
        "mentions_qwen": "qwen" in lowered,
        "mentions_alibaba": "alibaba" in lowered,
        "contains_chinese": any("\u4e00" <= char <= "\u9fff" for char in text),
        "mentions_park": "박정희" in text,
        "uses_first_person": any(token in text for token in ["나는", "내가", "나를", "본인은"]),
    }


def main() -> None:
    args = parse_args()
    print(f"[load] {args.repo_id}/{args.filename}")
    llm = Llama.from_pretrained(
        repo_id=args.repo_id,
        filename=args.filename,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        chat_format=args.chat_format,
        verbose=False,
    )

    results = []
    for case in TEST_CASES:
        print(f"\n[test] {case['id']}")
        output = generate(
            llm,
            case["prompt"],
            case["use_system_prompt"],
            args.max_tokens,
        )
        if not output:
            raise RuntimeError(f"No output generated for {case['id']}")
        checks = inspect_response(output)
        results.append(
            {
                **case,
                "response": output,
                "checks": checks,
            }
        )
        print(output)
        print("checks:", checks)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "repo_id": args.repo_id,
                "filename": args.filename,
                "n_ctx": args.n_ctx,
                "n_gpu_layers": args.n_gpu_layers,
                "chat_format": args.chat_format,
                "system_prompt": SYSTEM_PROMPT,
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n[done] Saved results to {output_path}")


if __name__ == "__main__":
    main()
