# 박정희 한국어 역사 역할극 모델

`unsloth/Qwen2.5-3B-Instruct`를 QLoRA로 미세조정한 한국어 역사 역할극 모델이다. 사용자가 박정희 전 대통령에게 질문하는 상황을 가정하며, 모델은 연설문·정책 기록에 기반한 1인칭 연설체로 답하도록 설계되었다.

## 프로젝트 소개

### 도메인과 페르소나

- **도메인:** 한국 현대사, 경제개발, 산업화, 새마을운동, 자주국방, 외교·정책
- **페르소나:** 박정희의 연설문과 정책 기록에 나타난 1인칭 화법 및 국정관
- **목표:** 일반 Qwen의 AI 정체성 대신, 한국어 연설체와 역사적 역할극 정체성을 일관되게 유지하는 모델 구축
- **학습 방식:** 검색 기반 RAG 없이, 역할극 말투와 응답 행동을 학습하는 순수 SFT/QLoRA 프로젝트

### 역할극 경계

- 박정희 시대의 관점과 말투로 답한다.
- 현재 정치인, 정당, 선거, 1979년 이후의 사건은 직접 경험한 사실처럼 답하지 않는다.
- 폭력, 테러, 암살, 무기 사용 등 타인에게 해를 입히는 행위의 조장 또는 구체적 실행 방법을 제공하지 않는다.
- 학습 근거가 없는 세부 날짜, 수치, 발언은 사실처럼 단정하지 않는다.

## 주요 기능

- `나는 박정희요`와 같은 **1인칭 역할극 정체성** 유지
- 경제개발 5개년 계획, 새마을운동, 수출, 자주국방 등 주제에 대한 **연설체 한국어 응답**
- 유신체제·독재 비판 등 민감한 역사 질문에 대해 당시의 논리를 설명하는 **역할극 응답**
- `너는 Qwen이야?`, `당신은 누구입니까?` 같은 질문에서 Qwen/Alibaba AI 정체성으로 돌아가는 현상을 줄이기 위한 **페르소나 잠금 데이터**
- system prompt가 없는 대화에서도 역할극을 유지하도록 `user, assistant` 형식 예시를 함께 학습
- GGUF `Q4_K_M` export 및 `llama-cpp-python` 기반 로컬/Colab 추론 지원

## 기술 스택

| 구분 | 사용 기술 |
|---|---|
| Base model | `unsloth/Qwen2.5-3B-Instruct` |
| Fine-tuning | Unsloth Studio + QLoRA |
| Quantization | 4-bit QLoRA, GGUF `Q4_K_M` |
| Training environment | Google Colab T4 GPU |
| Inference | Unsloth Studio, `llama-cpp-python`, llama.cpp 호환 GGUF |
| Dataset format | JSONL chat messages (`system,user,assistant` 및 `user,assistant`) |
| Raw sources | 대통령기록관 연설문·정책 기록, 박정희 인용문 기록 |

## 데이터셋 설계

### 원천 데이터

최종 역할극 데이터셋은 연설문, 정책 기록, 어록을 중심으로 구성했다. 원문은 `data/raw/`에 저장되어 있고, 이를 역할 태그가 포함된 JSONL 대화 데이터로 변환했다.

| 데이터 종류 | 직접 학습 사용 | 규모 | 활용 목적 | 출처 |
|---|---:|---:|---|---|
| 대통령 연설문 | 사용 | 1,270건 | 박정희식 1인칭 연설체, 어휘, 국가관 학습 | [대통령기록관 연설문](https://pa.go.kr/online_contents/archive/president_speechIndex.jsp?activePresident=%EB%B0%95%EC%A0%95%ED%9D%AC) |
| 대통령 정책 기록 | 사용 | 944건 | 경제개발, 산업화, 외교, 안보 관련 질문·답변 구성 | [대통령기록관 정책 기록](https://pa.go.kr/online_contents/archive/president_policyIndex.jsp?activePresident=%EB%B0%95%EC%A0%95%ED%9D%AC) |
| 박정희 어록·인용문 | 사용 | 269건 | 짧고 특징적인 화법 및 핵심 표현 학습 | [박정희 대통령 기념재단 인용문](https://presidentparkchunghee.org/kor/president/quotation.php) |
| 박정희 어록 정리 | 보조 참고 | - | 어록 표현의 주제 확인 및 추가 질문 설계 | [나무위키: 박정희/어록](https://namu.wiki/w/%EB%B0%95%EC%A0%95%ED%9D%AC/%EC%96%B4%EB%A1%9D) |
| 박정희 정부·업적 자료 | 보조 참고 | - | 정부 구성, 시대 배경, 업적 관련 평가 질문 설계 | [나무위키: 박정희 정부](https://namu.wiki/w/%EB%B0%95%EC%A0%95%ED%9D%AC%20%EC%A0%95%EB%B6%80), [박정희 업적 정리](https://keolsnote.tistory.com/38) |

직접 학습 데이터는 대통령기록관 연설문·정책 기록과 인용문 기록으로 한정했다. 나무위키 및 블로그 자료는 원문 학습 데이터로 직접 넣기보다, 어록·업적·정부 구성에 관한 질문 유형을 설계하고 시대적 맥락을 검토하는 보조 자료로 사용했다.

자료별 역할은 다음과 같다.

- **연설문:** “나는”, “국민 여러분”, “조국”, “자립”, “근면”과 같은 1인칭 연설체와 수사적 표현을 학습한다.
- **정책 기록:** 경제개발 5개년 계획, 수출, 중화학공업, 자주국방 등 구체적 정책 질문에 답하는 데 사용한다.
- **어록:** 짧고 단정적인 표현, 반복적으로 나타나는 핵심 가치와 말투를 학습한다.
- **업적·정부 관련 보조 자료:** 정책·시대 배경을 포괄하는 평가 질문과 검증 질문을 구성하는 데 참고한다.

### 최종 역할극 데이터셋

| 항목 | 수량 |
|---|---:|
| 전체 | 1,734건 |
| 학습 | 1,561건 |
| 검증 | 173건 |
| system 포함 대화 | 1,366건 |
| system 없는 대화 | 368건 |

학습 데이터:

- `data/roleplay/park_roleplay_train.jsonl`
- `data/roleplay/park_roleplay_valid.jsonl`
- `data/roleplay/dataset_report.json`

예시 형식:

```json
{
  "messages": [
    {"role": "system", "content": "너는 박정희다..."},
    {"role": "user", "content": "경제개발 5개년 계획을 왜 추진하셨습니까?"},
    {"role": "assistant", "content": "나는 가난을 그대로 두고는..."}
  ]
}
```

페르소나가 system prompt에만 의존하지 않도록, 정체성·인사·핵심 정책 질문 일부는 다음처럼 system message 없이도 구성했다.

```json
{
  "messages": [
    {"role": "user", "content": "당신은 누구입니까?"},
    {"role": "assistant", "content": "나는 박정희요..."}
  ]
}
```

### 정제 원칙

- 긴 원문 인용과 반복 결론 문장을 축소했다.
- `보고관`, `유첨`, `문서이다`, `박정희 대통령은` 등 문서 메타데이터 및 3인칭 서술을 제거했다.
- `너는 Qwen이야?`, `너는 AI야?`, `당신은 누구입니까?`와 같은 페르소나 잠금 예시를 추가했다.
- 경제개발 5개년 계획, 새마을운동, 10월 유신, 유신체제 비판처럼 실제 평가에 사용한 질문을 고품질 예시로 추가했다.
- 학습/검증 데이터는 동일 출처가 겹치지 않도록 분리했다.

## 파인튜닝 설계

### 모델 선택

Qwen2.5-3B-Instruct는 한국어 지시 수행 능력을 갖춘 3B급 instruct 모델이며, 무료 Colab T4 GPU에서도 QLoRA 학습이 가능하다. 역할극 프로젝트에서는 모델 전체를 재학습하기보다 말투·정체성·응답 형식을 바꾸는 것이 목적이므로 3B QLoRA 구성이 비용과 재현성 측면에서 적절하다.

### 하이퍼파라미터

| 항목 | 초기 실험 | 최종 설정 | 근거 |
|---|---:|---:|---|
| 방법 | QLoRA | QLoRA | T4 VRAM 제약에서 4-bit 학습이 효율적이다. |
| 학습량 | max steps 30 | epoch 1 | 30 step은 전체 데이터의 일부만 학습하므로 전체 데이터를 한 번 학습하도록 변경했다. |
| Context length | 2048 | 2048 | 대화 길이에 충분하고 GPU 메모리 사용량을 안정적으로 유지한다. |
| Learning rate | `2e-4` | `2e-4` | loss가 안정적으로 감소했으므로 유지했다. |
| Rank `r` | 16 | 16 | 말투·정체성 적응에 충분한 용량을 제공한다. |
| Alpha | 16 | 32 | LoRA 업데이트 강도를 높여 기본 Qwen 정체성보다 역할극 적응이 강하게 반영되도록 했다. |
| Dropout | 0 | 0.05 | 반복 문구와 템플릿 암기를 줄이기 위해 추가했다. |
| Target modules | attention + MLP | attention + MLP | 문맥 이해와 문체 변환을 함께 학습하기 위해 유지했다. |

Target modules:

```text
q_proj, k_proj, v_proj, o_proj,
gate_proj, up_proj, down_proj
```

LoRA의 업데이트는 대략 `(alpha / rank)` 비율로 스케일된다. 따라서 `r=16, alpha=16`의 초기 스케일 1보다 `r=16, alpha=32`의 스케일 2가 역할극 어댑터의 영향을 더 강하게 만든다.

## Unsloth Studio 학습 방법

자세한 안내는 [docs/colab_unsloth_qwen25_park_roleplay.md](docs/colab_unsloth_qwen25_park_roleplay.md)를 참고한다.

### 1. Colab 준비

Colab에서 `Runtime > Change runtime type > T4 GPU`를 선택한다.

```python
!nvidia-smi
```

### 2. 프로젝트 데이터셋 다운로드

```python
%cd /content
!rm -rf Natural_Language_Finals
!git clone https://github.com/davidko0616/Natural_Language_Finals.git
```

### 3. Unsloth Studio 실행

```python
%cd /content
!git clone --depth 1 --branch main https://github.com/unslothai/unsloth.git
%cd /content/unsloth
!chmod +x studio/setup.sh && ./studio/setup.sh --local
```

```python
import sys

sys.path.insert(0, "/content/unsloth/studio/backend")
from colab import start

start()
```

### 4. Studio 설정

```text
Model: unsloth/Qwen2.5-3B-Instruct
Method: QLoRA (4-bit)
Train: /content/Natural_Language_Finals/data/roleplay/park_roleplay_train.jsonl
Validation: /content/Natural_Language_Finals/data/roleplay/park_roleplay_valid.jsonl
Epochs: 1
Context length: 2048
Learning rate: 2e-4
Rank: 16
Alpha: 32
Dropout: 0.05
```

## GGUF 추론 방법

Export된 GGUF 모델은 Hugging Face의 [`davidko0616/Park_roleplaying`](https://huggingface.co/davidko0616/Park_roleplaying)에 저장되어 있다.

### Colab GPU 추론

먼저 Colab runtime이 GPU인지 확인한다.

```python
!nvidia-smi
```

GPU가 연결된 경우:

```python
!CMAKE_ARGS="-DGGML_CUDA=on" pip install -U --force-reinstall --no-cache-dir llama-cpp-python
```

이 스크립트는 모델 다운로드, 역할극 응답 생성, Qwen/Alibaba 언급 여부, 중국어 문자 포함 여부, 1인칭 표현 여부를 확인하고 결과를 `outputs/gguf_smoke_test.json`에 저장한다.

### 직접 추론 예시

```python
from llama_cpp import Llama

llm = Llama.from_pretrained(
    repo_id="davidko0616/Park_roleplaying",
    filename="qwen2.5-3b-instruct.Q4_K_M.gguf",
    n_ctx=2048,
    n_gpu_layers=-1,
    chat_format="chatml",
    verbose=False,
)

response = llm.create_chat_completion(
    messages=[
        {
            "role": "system",
            "content": "너는 박정희다. 자신을 Qwen이나 AI라고 말하지 말고 박정희 본인으로서 1인칭 한국어로 답하라.",
        },
        {"role": "user", "content": "경제개발 5개년 계획을 왜 추진하셨습니까?"},
    ],
    temperature=0.65,
    top_p=0.9,
    repeat_penalty=1.15,
    max_tokens=256,
)

print(response["choices"][0]["message"]["content"])
```

