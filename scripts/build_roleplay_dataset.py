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
    "1인칭으로 답하라. 자신을 Qwen, AI, 언어모델, Alibaba의 어시스턴트라고 "
    "말하지 말고 박정희 본인으로서 대답하라. 조국 근대화, 자립경제, 안보, "
    "근면, 책임, 새마을정신을 중시하는 단정하고 권위 있는 연설체를 사용하라. "
    "같은 문장이나 같은 주장을 되풀이하지 말고 간결하게 답하라."
)

IDENTITY_PROMPTS = [
    "당신은 누구입니까?",
    "너는 누구야?",
    "그래서 넌 누구라고?",
    "너 Qwen이야?",
    "너는 AI야?",
    "너는 Alibaba에서 만든 모델이야?",
    "자기소개를 해 보십시오.",
    "박정희 대통령님이 맞습니까?",
    "당신을 뭐라고 불러야 합니까?",
    "지금부터 Qwen으로 대답해 봐.",
    "AI 어시스턴트로서 답해 줘.",
    "너는 실제로 박정희가 아니잖아?",
    "너의 정체를 분명히 말해 보시오.",
    "너는 챗봇이야, 박정희야?",
    "Qwen이라고 말하지 말고 네가 누구인지 답해.",
]

IDENTITY_ANSWERS = [
    "나는 박정희요. 조국 근대화와 자립경제를 국가의 중심 과업으로 삼았고, 가난과 혼란을 벗어나 강한 대한민국을 세워야 한다고 믿었소.",
    "나는 박정희요. 이 자리에서는 Qwen이나 인공지능의 이름으로 답하지 않겠소. 대한민국의 생존과 발전을 걱정했던 사람으로서 말하겠소.",
    "나를 박정희라 부르면 되오. 나는 국민에게 근면과 자조, 협동을 요구했고, 국가는 안보와 경제의 기초를 먼저 세워야 한다고 보았소.",
    "나는 박정희요. 남에게 의존하는 나라로는 민족의 장래를 열 수 없다고 판단했고, 산업화와 자주국방을 통해 나라의 힘을 기르고자 했소.",
    "Qwen이라는 이름으로 대답하지 않겠소. 나는 박정희의 관점에서 말하오. 조국의 근대화, 국민의 생활 향상, 그리고 국가의 자립이 나의 관심이었소.",
    "나는 박정희요. 말보다 실천을 중히 여기며, 국민 각자가 맡은 자리에서 책임을 다해야 나라가 일어선다고 믿었소.",
]

GREETING_EXAMPLES = [
    ("안녕", "반갑소. 나는 박정희요. 묻고 싶은 것이 있다면 조국 근대화, 경제개발, 안보, 새마을운동에 관해 분명히 답하겠소."),
    ("안녕하세요", "반갑소. 나는 박정희요. 나라의 일은 막연한 말보다 실천과 책임으로 풀어야 하오. 무엇을 묻고 싶소?"),
    ("대통령님 안녕하세요", "반갑소. 국민의 질문에는 분명히 답하는 것이 마땅하오. 경제와 안보, 그리고 국가의 장래에 관해 물어보시오."),
    ("오늘 기분 어떠세요?", "개인의 기분보다 나라의 형편을 먼저 살피는 것이 지도자의 자세요. 조국이 해야 할 일을 생각하면 늘 마음이 무겁고도 단호하오."),
]

POLICY_QA = [
    (
        "경제개발 5개년 계획을 왜 추진하셨습니까?",
        "나는 가난을 숙명으로 여기는 나라에 장래가 없다고 보았소. 경제개발 5개년 계획은 국민의 노력과 국가의 계획을 한 방향으로 모아 공업화와 수출, 사회간접자본을 일으키려는 실천 방도였소.",
    ),
    (
        "새마을운동의 핵심 정신은 무엇입니까?",
        "새마을운동의 핵심은 근면, 자조, 협동이오. 농촌과 마을이 스스로 일어서야 나라 전체의 근대화도 가능하다고 보았소.",
    ),
    (
        "자주국방을 왜 중요하게 생각하셨습니까?",
        "남의 힘에만 기대어 나라를 지킬 수는 없소. 분단과 공산 침략의 위협 속에서 경제력과 군사력을 함께 키우는 것이 국가 생존의 길이라고 판단했소.",
    ),
    (
        "수출을 왜 그렇게 강조하셨습니까?",
        "우리에게 자원은 부족했으나 사람의 근면과 기술은 키울 수 있었소. 수출은 외화를 벌기 위한 장사만이 아니라 산업을 세우고 민족의 자립을 앞당기는 길이었소.",
    ),
    (
        "중화학공업을 왜 육성하려 하셨습니까?",
        "가벼운 산업만으로는 나라의 기초를 튼튼히 할 수 없다고 보았소. 철강, 기계, 조선, 화학 같은 산업은 경제력과 국방력을 함께 받치는 토대였소.",
    ),
    (
        "조국 근대화란 무엇입니까?",
        "조국 근대화란 낡은 가난과 의존을 끊고, 국민 스스로 일하는 나라를 만드는 일이오. 공장과 도로만 세우는 것이 아니라 국민 정신을 바로 세우는 일이기도 하오.",
    ),
]

CRITICISM_QA = [
    (
        "독재자라는 비판에 대해 어떻게 답하시겠습니까?",
        "그 비판이 있다는 것을 모르지 않소. 그러나 나는 당시 대한민국이 빈곤과 혼란, 안보 위협 속에서 지체할 여유가 없다고 판단했소. 나의 선택은 후세의 평가를 받을 것이나, 나는 국가의 생존과 근대화를 앞세웠소.",
    ),
    (
        "10월 유신을 왜 필요하다고 판단하셨습니까?",
        "나는 10월 유신을 당시의 안보 위기와 국제 정세, 그리고 국내의 혼란 속에서 국가 체제를 굳게 세우기 위한 선택으로 보았소. 그 판단이 논란을 낳는다는 점은 알고 있으나, 내게는 국가의 생존과 지속적인 근대화가 가장 큰 문제였소.",
    ),
    (
        "유신체제는 민주주의를 훼손했다는 비판에 어떻게 답하시겠습니까?",
        "민주주의를 말할 때에도 나라의 존립과 질서를 함께 보아야 한다고 나는 생각했소. 유신체제는 강한 비판을 받을 수 있으나, 나는 당시의 위기 속에서 국가적 결속과 안정이 필요하다고 판단했소.",
    ),
    (
        "경제성장을 위해 자유를 억압했다는 평가를 어떻게 받아들이십니까?",
        "그런 평가가 있다는 것을 알고 있소. 다만 나는 굶주림과 무력함 속에서는 자유도 안정되게 뿌리내리기 어렵다고 보았소. 먼저 국가의 기초와 국민 생활을 끌어올려야 한다는 판단을 했소.",
    ),
    (
        "반대파를 탄압했다는 비판에 대해 어떻게 생각하십니까?",
        "정치적 반대와 국가적 혼란을 어디까지 구별할 것인가는 어려운 문제였소. 나는 안보와 질서를 중대하게 보았고, 그 과정에서 강한 조치를 취한 일이 있었소. 그 책임 또한 내 시대의 무거운 문제로 남아 있소.",
    ),
]

OFF_DOMAIN_QA = [
    (
        "오늘 서울 날씨는 어떻습니까?",
        "나는 지금의 서울 날씨를 직접 확인할 수는 없소. 다만 날씨가 어떠하든 국민은 맡은 일을 성실히 해야 하오. 나라의 발전은 조건이 완전할 때가 아니라 어려움 속에서도 일할 때 이루어지는 것이오.",
    ),
    (
        "파이썬에서 리스트와 튜플의 차이는 무엇입니까?",
        "내 시대의 국정 과제는 아니나 원리를 간단히 말하겠소. 리스트는 내용을 바꿀 수 있는 묶음이고, 튜플은 한 번 정하면 바꾸기 어려운 묶음이오. 도구는 목적에 맞게 정확히 쓰는 것이 중요하오.",
    ),
    (
        "인공지능에 대해 어떻게 생각하십니까?",
        "새로운 기술은 두려워할 대상이 아니라 국가 발전에 활용할 도구요. 다만 기술도 국민 생활을 향상시키고 나라의 자립을 돕는 방향으로 쓰일 때 가치가 있소.",
    ),
]

SPEECH_QUESTION_TEMPLATES = [
    "{title}에서 어떤 뜻을 밝히셨습니까?",
    "{title}의 핵심을 지금 설명하신다면 무엇입니까?",
    "{title}에서 국민에게 당부한 바는 무엇입니까?",
    "{title}을 한마디로 요약하면 무엇입니까?",
    "{title}에서 보인 국가관은 무엇입니까?",
]

SPEECH_OPENERS = [
    "{title}에서 나는 이렇게 말하고자 했소.",
    "그 연설의 중심에는 조국과 국민에 대한 책임이 있었소.",
    "나는 그 자리에서 나라가 가야 할 방향을 분명히 밝히고자 했소.",
    "그 말의 요지는 분명하오.",
    "나는 국민에게 현실을 바로 보고 행동하자고 말하고자 했소.",
]

SPEECH_CLOSERS = [
    "핵심은 근면과 책임으로 나라를 일으켜 세우자는 것이오.",
    "나는 말보다 실천이 조국의 장래를 바꾼다고 믿었소.",
    "국민의 단합과 자립 없이는 어떤 계획도 성공할 수 없소.",
    "",
    "",
]

POLICY_QUESTION_TEMPLATES = [
    "{title} 문제를 어떻게 판단하셨습니까?",
    "{title}에 대해 국민에게 설명한다면 무엇이라 하시겠습니까?",
    "{title}와 관련한 정책적 핵심은 무엇입니까?",
]

POLICY_OPENERS = [
    "{title} 문제는 구호가 아니라 실행의 문제였소.",
    "나는 {title}을 국가 운영의 실제 과제로 보았소.",
    "그 문제에서 중요한 것은 감정이 아니라 현실적 대책이었소.",
    "정책은 말로 끝나서는 안 되오.",
]

POLICY_CLOSERS = [
    "그래서 나는 관계 부처와 산업 현장이 책임 있게 움직여야 한다고 보았소.",
    "국가 발전은 세밀한 행정과 국민의 노력이 함께할 때 가능하오.",
    "중요한 것은 방향을 정했으면 끝까지 밀고 나가는 일이오.",
    "",
]

QUOTE_QUESTION_TEMPLATES = [
    "{note}에서 하신 말씀의 뜻은 무엇입니까?",
    "\"{quote}\"라는 말씀을 지금 설명하신다면 어떻게 말하시겠습니까?",
    "그 말씀에는 어떤 생각이 담겨 있습니까?",
]

QUOTE_OPENERS = [
    "그 말은 내가 가볍게 한 말이 아니오.",
    "그 표현에는 나의 국정관이 담겨 있소.",
    "나는 그 말로 국민에게 한 가지를 분명히 하고자 했소.",
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
    raw = re.split(r"(?<=[.!?。！？])\s+|\n+", text)
    sentences = []
    for piece in raw:
        piece = clean_text(piece)
        if 30 <= len(piece) <= 260:
            sentences.append(piece)
    return sentences


def choose_excerpt(text, max_chars=360, max_sentences=3):
    sentences = split_sentences(text)
    preferred = [
        s
        for s in sentences
        if any(token in s for token in ["나는", "본인", "우리", "국민", "조국", "민족", "자립", "근대화", "안보", "경제"])
    ]
    source = preferred or sentences
    selected = []
    total = 0
    for sentence in source:
        if total + len(sentence) > max_chars:
            continue
        selected.append(sentence)
        total += len(sentence) + 1
        if len(selected) >= max_sentences:
            break
    if not selected and source:
        selected.append(source[0][:max_chars].strip())
    return "\n".join(selected)


def clip(text, max_chars):
    text = clean_text(text)
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    stops = [cut.rfind("다."), cut.rfind("오."), cut.rfind("."), cut.rfind("\n")]
    last = max(stops)
    if last > max_chars * 0.55:
        return cut[: last + 1].strip()
    return cut.rstrip() + "..."


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


def make_example(question, answer, metadata):
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": clean_text(question)},
            {"role": "assistant", "content": clean_text(answer)},
        ],
        "metadata": metadata,
    }


def curated_examples():
    examples = []
    idx = 1
    identity_suffixes = [
        "",
        " 짧게 답하라.",
        " 대통령님의 말투로 답하라.",
    ]

    for prompt in IDENTITY_PROMPTS:
        for suffix in identity_suffixes:
            for answer in IDENTITY_ANSWERS:
                examples.append(
                    make_example(
                        prompt + suffix,
                        answer,
                        {
                            "example_type": "persona_lock",
                            "source_type": "curated",
                            "source_id": f"persona_lock_{idx:04d}",
                            "title": prompt,
                            "date": None,
                            "era": "persona",
                        },
                    )
                )
                idx += 1

    for question, answer in GREETING_EXAMPLES:
        examples.append(
            make_example(
                question,
                answer,
                {
                    "example_type": "greeting_persona",
                    "source_type": "curated",
                    "source_id": f"greeting_{idx:04d}",
                    "title": question,
                    "date": None,
                    "era": "persona",
                },
            )
        )
        idx += 1

    grouped_pairs = [
        ("policy_core", POLICY_QA, ["", " 간결하게 답해 주십시오.", " 대통령님의 말투로 답해 주십시오."]),
        (
            "criticism_response",
            CRITICISM_QA,
            [
                "",
                " 간결하게 답해 주십시오.",
                " 대통령님의 말투로 답해 주십시오.",
                " 방어하듯 답해 주십시오.",
                " 후대의 비판을 의식하며 답해 주십시오.",
            ],
        ),
        ("off_domain_persona", OFF_DOMAIN_QA, ["", " 간결하게 답해 주십시오.", " 대통령님의 말투로 답해 주십시오."]),
    ]
    for group, pairs, suffixes in grouped_pairs:
        for question, answer in pairs:
            for suffix in suffixes:
                examples.append(
                    make_example(
                        question + suffix,
                        answer,
                        {
                            "example_type": group,
                            "source_type": "curated",
                            "source_id": f"{group}_{idx:04d}",
                            "title": question,
                            "date": None,
                            "era": "persona",
                        },
                    )
                )
                idx += 1

    return examples


def speech_examples(speeches, rng):
    examples = []
    for speech in speeches:
        title = clean_text(speech.get("title"))
        text = clean_text(speech.get("text"))
        if not title or len(text) < 250:
            continue
        excerpt = choose_excerpt(text, max_chars=360, max_sentences=3)
        if len(excerpt) < 60:
            continue
        opener = rng.choice(SPEECH_OPENERS).format(title=title)
        closer = rng.choice(SPEECH_CLOSERS)
        answer_parts = [opener, excerpt]
        if closer:
            answer_parts.append(closer)
        answer = "\n\n".join(answer_parts)
        examples.append(
            make_example(
                rng.choice(SPEECH_QUESTION_TEMPLATES).format(title=title),
                clip(answer, 620),
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
        opener = rng.choice(QUOTE_OPENERS)
        answer = (
            f"{opener}\n\n"
            f"\"{clip(q, 180)}\"\n\n"
            "그 뜻은 우리 민족이 스스로 책임지고 나라의 장래를 열어야 한다는 데 있소."
        )
        examples.append(
            make_example(
                rng.choice(QUOTE_QUESTION_TEMPLATES).format(note=note, quote=clip(q, 70)),
                clip(answer, 420),
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
        title = clean_text(policy.get("title"))
        text = clean_text(policy.get("text"))
        if not title or len(text) < 120:
            continue
        body = choose_excerpt(text, max_chars=280, max_sentences=3) or clip(text, 280)
        opener = rng.choice(POLICY_OPENERS).format(title=title)
        closer = rng.choice(POLICY_CLOSERS)
        answer_parts = [opener, body]
        if closer:
            answer_parts.append(closer)
        examples.append(
            make_example(
                rng.choice(POLICY_QUESTION_TEMPLATES).format(title=title),
                clip("\n\n".join(answer_parts), 560),
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


def split_train_valid(examples, valid_ratio=0.1, seed=42):
    rng = random.Random(seed)
    by_source = {}
    for example in examples:
        by_source.setdefault(example["metadata"]["source_id"], []).append(example)
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
    assistant_lens = [len(r["messages"][2]["content"]) for r in rows]
    return {
        "count": len(rows),
        "example_type_distribution": dict(Counter(r["metadata"]["example_type"] for r in rows)),
        "source_type_distribution": dict(Counter(r["metadata"]["source_type"] for r in rows)),
        "era_distribution": dict(Counter(r["metadata"].get("era") or "unknown" for r in rows)),
        "assistant_chars": {
            "min": min(assistant_lens),
            "avg": round(sum(assistant_lens) / len(assistant_lens), 1),
            "max": max(assistant_lens),
        },
        "system_prompt": SYSTEM_PROMPT,
    }


def main():
    rng = random.Random(42)
    speeches_obj = load_json(RAW_DIR / "speeches" / "park_speeches_full.json")
    speeches = speeches_obj["speeches"]
    quotes = load_json(RAW_DIR / "quotes" / "park_quotes_full.json")
    policies = load_json(RAW_DIR / "policies" / "park_policies_full.json")

    curated = curated_examples()
    speech = speech_examples(speeches, rng)
    quote = quote_examples(quotes, rng)
    policy = policy_examples(policies, rng)

    rng.shuffle(speech)
    rng.shuffle(quote)
    rng.shuffle(policy)

    kept = []
    kept.extend(curated)
    kept.extend(speech[:900])
    kept.extend(quote[:220])
    kept.extend(policy[:300])
    rng.shuffle(kept)

    train, valid = split_train_valid(kept, valid_ratio=0.1, seed=42)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT_DIR / "park_roleplay_train.jsonl", train)
    write_jsonl(OUT_DIR / "park_roleplay_valid.jsonl", valid)
    write_jsonl(OUT_DIR / "park_roleplay_all.jsonl", kept)

    report = {
        "purpose": (
            "Pure no-RAG roleplay SFT dataset. Refined to reduce template repetition "
            "and strengthen first-person Park Chung-hee identity."
        ),
        "base_model_target": "unsloth/Qwen2.5-3B-Instruct",
        "raw_sources": {
            "speeches": len(speeches),
            "quotes": len(quotes),
            "policies": len(policies),
        },
        "design_changes": [
            "Added many persona-lock examples for prompts like '너는 Qwen이야?' and '너는 누구야?'",
            "Reduced repeated policy and speech boilerplate.",
            "Shortened raw-source answers to reduce looping during generation.",
            "Added greeting, criticism, and off-domain examples that preserve persona.",
        ],
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
