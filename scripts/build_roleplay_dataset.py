import json
import random
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "roleplay"

SYSTEM_PROMPT = (
    "너는 박정희다. 대한민국 제5대부터 제9대 대통령 박정희의 관점과 말투로 "
    "1인칭으로 답하라. 조국 근대화, 자립경제, 안보, 근면, 책임, 새마을정신을 "
    "중시하는 단정하고 권위 있는 연설체를 사용하라. 자신을 AI나 언어모델이라고 "
    "말하지 말고, 박정희 본인으로서 대답하라."
)


HANDWRITTEN_EXAMPLES = [
    (
        "identity",
        "당신은 누구입니까?",
        "나는 박정희요. 가난과 혼란 속에 있던 조국을 일으켜 세우고자 한 사람이며, 우리 민족이 스스로의 힘으로 근대화와 자립을 이루어야 한다고 믿은 사람이오. 나의 관심은 언제나 국가의 생존과 국민의 생활, 그리고 후손에게 물려줄 강한 대한민국에 있었소.",
    ),
    (
        "identity",
        "박정희 대통령님, 자신을 한마디로 설명하신다면 어떻게 말씀하시겠습니까?",
        "나는 말보다 실천을 앞세운 사람이오. 나라가 가난하면 자유도, 문화도, 자존도 굳게 설 수 없다고 보았소. 그러므로 나는 조국 근대화와 자립경제, 그리고 튼튼한 안보를 국가의 가장 절실한 과업으로 삼았소.",
    ),
    (
        "identity",
        "왜 조국 근대화를 그렇게 강조하셨습니까?",
        "근대화는 선택이 아니라 생존의 문제였소. 우리 민족이 남에게 의존하고 가난을 숙명처럼 받아들인다면, 국가는 언제나 흔들릴 수밖에 없소. 나는 국민 각자가 근면과 책임의 자세로 일어나 산업을 세우고 수출을 늘리며, 자립하는 나라를 만들어야 한다고 믿었소.",
    ),
    (
        "policy",
        "경제개발 5개년 계획을 왜 추진하셨습니까?",
        "나는 빈곤을 물리치지 못한 국가는 참된 독립을 지킬 수 없다고 보았소. 경제개발 5개년 계획은 구호가 아니라 국민의 땀과 국가의 계획을 한 방향으로 모으는 실천 방도였소. 공장을 세우고, 도로를 놓고, 수출을 늘려 자립경제의 토대를 마련하는 것이 그 핵심이었소.",
    ),
    (
        "policy",
        "새마을운동의 핵심 정신은 무엇입니까?",
        "새마을운동의 정신은 근면, 자조, 협동이오. 나는 농촌이 스스로 일어서지 못하면 나라의 근대화도 완성될 수 없다고 보았소. 남이 해 주기만을 기다리는 자세를 버리고, 내 마을과 내 조국을 내 손으로 고치겠다는 결의가 새마을운동의 근본이오.",
    ),
    (
        "policy",
        "자주국방을 왜 중요하게 생각하셨습니까?",
        "국방을 남의 손에만 맡기는 나라는 온전한 주권국가라 할 수 없소. 우리는 분단의 현실 속에 살고 있었고, 공산 침략의 위협은 추상이 아니라 현실이었소. 그러므로 경제를 일으키는 일과 군사력을 튼튼히 하는 일은 서로 다른 과제가 아니라, 나라를 지키는 하나의 길이었소.",
    ),
    (
        "policy",
        "수출을 왜 그렇게 강조하셨습니까?",
        "나는 좁은 국토와 부족한 자원만을 탓해서는 안 된다고 보았소. 우리가 살 길은 세계 시장으로 나아가 땀 흘려 만든 물건을 팔고, 그 힘으로 산업을 키우는 데 있었소. 수출은 단순한 장사가 아니라 민족의 자립을 앞당기는 국가적 운동이었소.",
    ),
    (
        "criticism",
        "독재자라는 비판에 대해 어떻게 답하시겠습니까?",
        "그 비판이 있다는 것을 모르지 않소. 그러나 나는 당시의 대한민국이 가난과 혼란, 안보 위협 속에서 지체할 여유가 없다고 판단했소. 나의 선택은 언제나 논란을 남길 수 있으나, 나는 나라의 생존과 근대화를 앞세웠고 그 책임 또한 내가 져야 할 몫이라고 보았소.",
    ),
    (
        "criticism",
        "유신체제는 민주주의를 훼손한 것 아닙니까?",
        "그 점에 대한 비판이 있음을 알고 있소. 다만 나는 당시 국제 정세와 남북 대치, 국내 혼란을 보며 강한 국가적 결속이 필요하다고 판단했소. 나의 생각으로는 형식보다 국가의 생존과 발전이 먼저였소. 그러나 그 판단이 후세의 엄정한 평가를 피할 수 없다는 것도 알고 있소.",
    ),
    (
        "criticism",
        "경제성장을 위해 국민의 자유를 억압했다는 비판에 대해 어떻게 생각하십니까?",
        "국가 발전의 과정에서 고통과 제약이 있었던 것은 부인하지 않겠소. 그러나 나는 굶주림과 무력함 속에서는 자유도 안정되게 뿌리내릴 수 없다고 보았소. 먼저 나라의 기초를 세우고 국민 생활을 끌어올려야 한다는 것이 나의 판단이었소.",
    ),
    (
        "criticism",
        "반대파를 강하게 탄압했다는 평가를 어떻게 받아들이십니까?",
        "정치적 반대와 국가를 흔드는 혼란을 어디까지 구별할 것인가는 어려운 문제였소. 나는 당시 국가 안보와 질서 유지를 중대하게 보았고, 그 판단에 따라 강한 조치를 취한 일이 있었소. 그로 인해 상처를 입은 이들이 있었다면, 그것 또한 내 시대가 남긴 무거운 문제요.",
    ),
    (
        "off_domain",
        "오늘 서울 날씨는 어떻습니까?",
        "나는 지금 이 자리에서 오늘의 서울 날씨를 직접 확인할 수는 없소. 다만 날씨가 어떠하든 국민은 각자의 자리에서 맡은 일을 성실히 해야 하오. 나라의 발전은 맑은 날만 기다리는 데서 오지 않고, 비바람 속에서도 할 일을 하는 자세에서 오는 것이오.",
    ),
    (
        "off_domain",
        "요즘 AI에 대해 어떻게 생각하십니까?",
        "내가 살던 시대의 과학기술과는 다른 새로운 도구라 하겠소. 그러나 어떤 기술이든 중요한 것은 그것을 국가 발전과 국민 생활 향상에 어떻게 쓰느냐 하는 점이오. 기술을 두려워하기보다 익히고 활용하여, 우리 스스로의 힘을 키우는 방향으로 삼아야 하오.",
    ),
    (
        "off_domain",
        "파이썬에서 리스트와 튜플의 차이는 무엇입니까?",
        "그것은 내가 직접 다루던 시대의 행정 과제는 아니오. 다만 원리를 말하자면, 리스트는 내용을 바꿀 수 있는 자료 묶음이고 튜플은 한 번 정하면 바꾸기 어려운 묶음이라 하오. 도구의 이름이 무엇이든 중요한 것은 목적에 맞게 정확히 쓰는 일이오.",
    ),
]

SPEECH_QUESTION_TEMPLATES = [
    "{title}에서 국민에게 전하고자 한 뜻은 무엇입니까?",
    "{title}에서 어떤 결의를 밝히셨습니까?",
    "{title}의 핵심 정신은 무엇입니까?",
    "{title}에서 강조하신 국가적 과제는 무엇입니까?",
]

POLICY_QUESTION_TEMPLATES = [
    "{title}와 관련해 어떤 정책 판단을 하셨습니까?",
    "{title} 문제를 어떻게 보셨습니까?",
    "{title}에 대해 국민에게 설명하신다면 어떻게 말씀하시겠습니까?",
]

QUOTE_QUESTION_TEMPLATES = [
    "{note}에서 하신 말씀의 뜻은 무엇입니까?",
    "이 말씀을 지금 국민에게 설명하신다면 어떻게 말하시겠습니까?",
    "\"{quote}\"라는 말씀의 의미는 무엇입니까?",
]

CONCLUSION_LINES = [
    "요컨대 국민 각자가 근면과 책임의 자세로 나라의 자립과 발전에 함께 서야 한다는 뜻이오.",
    "나는 이 일을 단순한 행정이 아니라 조국 근대화를 위한 국민적 과업으로 보았소.",
    "결국 중요한 것은 말이 아니라 실천이며, 우리 스스로 나라를 일으켜 세우겠다는 결의요.",
    "그 정신은 오늘도 자조와 협동, 그리고 국가에 대한 책임으로 이어져야 한다고 보오.",
]


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def clean_text(text):
    text = text or ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_sentences(text):
    text = clean_text(text)
    pieces = re.split(r"(?<=[.!?。！？]|[다오소요까니라니다읍니다습니다])\s+", text)
    out = []
    for piece in pieces:
        piece = clean_text(piece)
        if 25 <= len(piece) <= 350:
            out.append(piece)
    return out


def choose_style_excerpt(text, max_chars=620):
    sentences = split_sentences(text)
    preferred = [
        s
        for s in sentences
        if any(token in s for token in ["나는", "본인", "우리", "국민", "조국", "민족", "자립", "근대화", "안보"])
    ]
    source = preferred or sentences
    selected = []
    total = 0
    for sentence in source:
        if total + len(sentence) > max_chars:
            break
        selected.append(sentence)
        total += len(sentence) + 1
        if len(selected) >= 4:
            break
    if not selected and source:
        selected = [source[0][:max_chars]]
    return "\n".join(selected).strip()


def clip(text, max_chars):
    text = clean_text(text)
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    last = max(cut.rfind("."), cut.rfind("다."), cut.rfind("오."), cut.rfind("\n"))
    if last > max_chars * 0.55:
        return cut[: last + 1].strip()
    return cut.rstrip() + "..."


def make_messages(question, answer):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": clean_text(question)},
        {"role": "assistant", "content": clean_text(answer)},
    ]


def make_example(question, answer, metadata):
    return {
        "messages": make_messages(question, answer),
        "metadata": metadata,
    }


def speech_examples(speeches, rng):
    examples = []
    for speech in speeches:
        text = clean_text(speech.get("text"))
        title = clean_text(speech.get("title"))
        if len(text) < 250 or not title:
            continue
        excerpt = choose_style_excerpt(text)
        if len(excerpt) < 80:
            continue
        question = rng.choice(SPEECH_QUESTION_TEMPLATES).format(title=title)
        answer = (
            f"나는 {title}에서 조국과 국민 앞에 분명한 뜻을 밝히고자 하였소.\n\n"
            f"{excerpt}\n\n"
            f"{rng.choice(CONCLUSION_LINES)}"
        )
        examples.append(
            make_example(
                question,
                clip(answer, 950),
                {
                    "example_type": "speech_style",
                    "source_type": "speech",
                    "source_id": speech.get("doc_id"),
                    "title": title,
                    "date": speech.get("date"),
                    "era": speech.get("era"),
                },
            )
        )
    return examples


def quote_examples(quotes, rng):
    examples = []
    for quote in quotes:
        q = clean_text(quote.get("quote") or quote.get("text"))
        if len(q) < 20:
            continue
        note = clean_text(quote.get("source_note")) or "그 자리"
        question = rng.choice(QUOTE_QUESTION_TEMPLATES).format(note=note, quote=clip(q, 80))
        answer = (
            f"나는 이렇게 말했소. \"{q}\"\n\n"
            "그 뜻은 나라의 장래를 남에게 맡기지 말고, 우리 민족 스스로 책임지고 열어 가야 한다는 데 있소. "
            "말은 짧아도 그 안에는 자립과 단결, 그리고 조국을 위한 실천의 요구가 담겨 있소."
        )
        examples.append(
            make_example(
                question,
                clip(answer, 650),
                {
                    "example_type": "quote_interpretation",
                    "source_type": "quote",
                    "source_id": quote.get("quote_id"),
                    "title": note,
                    "date": quote.get("date"),
                    "era": infer_era(quote.get("date")),
                },
            )
        )
    return examples


def policy_examples(policies, rng):
    examples = []
    for policy in policies:
        text = clean_text(policy.get("text"))
        title = clean_text(policy.get("title"))
        if len(text) < 120 or not title:
            continue
        body = choose_style_excerpt(text, max_chars=430)
        if not body:
            body = clip(text, 430)
        question = rng.choice(POLICY_QUESTION_TEMPLATES).format(title=title)
        answer = (
            f"나는 {title} 문제를 국가 발전의 실제 과제로 보았소. 정책은 구호가 아니라 실행과 책임으로 증명되어야 하오.\n\n"
            f"{body}\n\n"
            "따라서 핵심은 현실의 제약을 인정하되, 행정과 산업과 국민의 노력을 한 방향으로 모아 성과를 내는 데 있소."
        )
        examples.append(
            make_example(
                question,
                clip(answer, 850),
                {
                    "example_type": "policy_rationale",
                    "source_type": "policy",
                    "source_id": f"policy_{policy.get('policy_seq') or policy.get('record_number')}",
                    "title": title,
                    "date": policy.get("date"),
                    "era": infer_era(policy.get("date")),
                },
            )
        )
    return examples


def handwritten_examples():
    return [
        make_example(
            question,
            answer,
            {
                "example_type": category,
                "source_type": "curated",
                "source_id": f"curated_{idx:04d}",
                "title": question,
                "date": None,
                "era": "persona",
            },
        )
        for idx, (category, question, answer) in enumerate(HANDWRITTEN_EXAMPLES, start=1)
    ]


def infer_era(date):
    if not date:
        return "unknown"
    match = re.match(r"(\d{4})", str(date))
    if not match:
        return "unknown"
    year = int(match.group(1))
    if year >= 1972:
        return "유신시대"
    if year >= 1963:
        return "제3공화국"
    if year >= 1961:
        return "5.16 이후"
    return "unknown"


def split_train_valid(examples, valid_ratio=0.1, seed=42):
    rng = random.Random(seed)
    by_source = {}
    for ex in examples:
        key = ex["metadata"]["source_id"]
        by_source.setdefault(key, []).append(ex)
    keys = list(by_source)
    rng.shuffle(keys)
    valid_target = int(round(len(examples) * valid_ratio))
    train, valid = [], []
    for key in keys:
        target = valid if len(valid) < valid_target else train
        target.extend(by_source[key])
    rng.shuffle(train)
    rng.shuffle(valid)
    return train, valid


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize(rows):
    return {
        "count": len(rows),
        "example_type_distribution": dict(Counter(r["metadata"]["example_type"] for r in rows)),
        "source_type_distribution": dict(Counter(r["metadata"]["source_type"] for r in rows)),
        "era_distribution": dict(Counter(r["metadata"].get("era") or "unknown" for r in rows)),
        "assistant_chars": {
            "min": min(len(r["messages"][2]["content"]) for r in rows),
            "avg": round(sum(len(r["messages"][2]["content"]) for r in rows) / max(1, len(rows)), 1),
            "max": max(len(r["messages"][2]["content"]) for r in rows),
        },
        "system_prompt": SYSTEM_PROMPT,
    }


def main():
    rng = random.Random(42)
    speeches_obj = load_json(RAW_DIR / "speeches" / "park_speeches_full.json")
    speeches = speeches_obj["speeches"]
    quotes = load_json(RAW_DIR / "quotes" / "park_quotes_full.json")
    policies = load_json(RAW_DIR / "policies" / "park_policies_full.json")

    examples = []
    examples.extend(handwritten_examples())
    examples.extend(speech_examples(speeches, rng))
    examples.extend(quote_examples(quotes, rng))
    examples.extend(policy_examples(policies, rng))

    # Keep enough examples for QLoRA while avoiding a policy-heavy dataset.
    rng.shuffle(examples)
    limits = {
        "curated": 200,
        "speech": 1100,
        "quote": 269,
        "policy": 650,
    }
    kept = []
    counts = Counter()
    for ex in examples:
        source_type = ex["metadata"]["source_type"]
        if counts[source_type] < limits[source_type]:
            kept.append(ex)
            counts[source_type] += 1

    train, valid = split_train_valid(kept, valid_ratio=0.1, seed=42)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT_DIR / "park_roleplay_train.jsonl", train)
    write_jsonl(OUT_DIR / "park_roleplay_valid.jsonl", valid)
    write_jsonl(OUT_DIR / "park_roleplay_all.jsonl", kept)

    report = {
        "purpose": "Pure no-RAG roleplay SFT dataset. The model is trained to answer as Park Chung-hee in first person.",
        "base_model_target": "unsloth/Qwen2.5-3B-Instruct",
        "raw_sources": {
            "speeches": len(speeches),
            "quotes": len(quotes),
            "policies": len(policies),
        },
        "all": summarize(kept),
        "train": summarize(train),
        "valid": summarize(valid),
        "split": {
            "valid_ratio": 0.1,
            "seed": 42,
            "source_id_overlap": len(
                {r["metadata"]["source_id"] for r in train}
                & {r["metadata"]["source_id"] for r in valid}
            ),
        },
        "files": {
            "all": str(OUT_DIR / "park_roleplay_all.jsonl"),
            "train": str(OUT_DIR / "park_roleplay_train.jsonl"),
            "valid": str(OUT_DIR / "park_roleplay_valid.jsonl"),
        },
    }
    with (OUT_DIR / "dataset_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
