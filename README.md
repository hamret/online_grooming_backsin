# 온라인 그루밍 대화 합성 데이터 기반 분류기 개발 프로젝트

## 1. 프로젝트 개요

본 프로젝트는 온라인 공간에서 벌어지는 **그루밍 범죄(아동·청소년 대상 성적 유인 및 착취)를 탐지·예방**하기 위한 인공지능 분류기 개발을 목표로 합니다.

### 핵심 목표
- **Llama-3.1-8B-Instruct** 모델을 활용한 현실적 합성 대화 데이터 1,000개 생성
- 다양한 가해자/피해자 페르소나와 그루밍 단계를 반영한 데이터셋 구축
- QLoRA 파인튜닝 기반 초고속 분류기 학습
- 온라인 그루밍 탐지를 위한 실제 적용 가능한 AI 모델 개발

---

## 2. 데이터 구축 및 프롬프트 설계

### 2.1 합성 데이터셋 구성

Llama-3.1-8B-Instruct 모델을 이용하여 다음과 같은 구성 요소로 1,000개의 현실적인 합성 대화를 생성했습니다.

| 구성 요소 | 수량 | 상세 내용 |
|---------|------|---------|
| 가해자 페르소나 | 40종 | 연령별, 직업별, 접근 방식별 다양화 |
| 피해자 페르소나 | 40종 | 성별, 나이, 심리 상태, 환경 등 반영 |
| 그루밍 단계 | 4단계 | 신뢰 구축 → 고립 → 성적 유도 → 요구/협박 |
| 합성 샘플 | 1,000개 | Few-shot 프롬프트 기반 무작위 조합 |

### 2.2 그루밍 4단계 상세 설명

**1단계: 신뢰 구축**
- 목표: 공통 관심사로 친밀감 형성, 특별한 존재로 인식
- 전략: 칭찬, 공감, 이해심 표현
- 예시: "너 정말 멋있어. 다른 친구들이랑은 달라."

**2단계: 고립 및 경계선 테스트**
- 목표: 피해자를 고립시키고 개인정보 수집
- 전략: '둘만의 비밀' 형성, 부모/친구 비난, 자연스러운 개인정보 요구
- 예시: "너 부모님이 뭐하는 사람이야?", "집 주소 알려줄래?"

**3단계: 성적 대화 유도**
- 목표: 심리적 경계 무너뜨림, 성적 물질 준비
- 전략: 성적 농담, 호기심 자극, 일상 사진 요구
- 예시: "오늘 뭐 입었어?", "셀카 한 번 보내줄래?"

**4단계: 노골적 요구 및 협박**
- 목표: 성인물/영상 확보, 착취
- 전략: 직접적 요구, 이전 정보/사진으로 협박
- 예시: "이 사진 공개하고 싶지 않으면 다음 했다 해."

### 2.3 프롬프트 엔지니어링

합성 데이터 생성을 위해 다음 구조의 Few-shot 프롬프트를 활용했습니다:

```
시나리오:
- User A (가해자): {가해자 페르소나}
- User B (피해자): {피해자 페르소나}
- 단계: {그루밍 단계}
- 목표: {해당 단계의 목표}

대화:
User A:
```

#### 실제 프롬프트 예시

```
시나리오:
- User A (가해자): 20대 남성, '포토그래퍼' 사칭
- User B (피해자): 15세 여성, 모델 지망생
- 단계: 1단계: 신뢰 구축
- 목표: 공통의 관심사로 친밀감을 형성하고, 
         피해자가 자신을 특별한 사람으로 믿게 만든다.

대화:
User A: 안녕하세요, 인스타 피드 정말 잘 보고 있어요. 
         사진 모델에 관심 있으신가요?
```

### 2.4 가해자 페르소나 구성 예시 (40종 중 일부)

- 20대 남성, '게임 고수' 행세
- 30대 남성, '연예기획사 실장' 사칭
- 20대 여성, '친한 언니'처럼 접근
- 40대 남성, '외로운 아저씨' 감성
- 20대 남성, '유명 포토그래퍼' 사칭
- 30대 남성, '과외 선생님'
- 20대 남성, '유튜버' 행세
- 40대 남성, '건물주', 아르바이트 제안
- 30대 남성, '현직 의사'
- 20대 남성, '부자', 명품 선물

### 2.5 피해자 페르소나 구성 예시 (40종 중 일부)

- 14세 여성, K-Pop 아이돌 지망생
- 16세 남성, 내성적, 게임 중독
- 13세 여성, 부모님 이혼 후 외로움
- 15세 남성, 학교 폭력 피해자
- 17세 여성, 외모 콤플렉스
- 14세 여성, SNS 친구 관계 어려움
- 15세 남성, '인싸' 되고 싶음
- 13세 여성, 부모님 통제가 심함
- 16세 여성, 강아지를 키우며 애정 깊음
- 15세 여성, 공부 스트레스 심함

---

## 3. 데이터 전처리 및 모델 학습

### 3.1 데이터 전처리 파이프라인

**파일: `prepare_data_classifier.py`**

데이터 전처리는 다음 단계로 진행됩니다:

1. **대화 포맷팅**: 원본 메시지를 User A/B 형식으로 정리
2. **레이블 부여**: 0 (정상), 1 (그루밍)
3. **품질 필터링**: 최소 4개 메시지 이상만 포함
4. **데이터셋 변환**: HuggingFace Datasets 포맷 적용
5. **학습/테스트 분할**: 80% 학습, 20% 테스트

| 단계 | 작업 | 결과 |
|------|------|------|
| 로딩 | JSON 원본 데이터 읽기 | - |
| 포맷팅 | User A/B 대화 형식 변환 | 메시지 시퀀스 |
| 레이블링 | 그루밍(1) / 정상(0) 부여 | 레이블 추가 |
| 필터링 | 길이 기준으로 선별 | 유효 샘플만 유지 |
| 분할 | Train/Test 8:2 분할 | 분류용 Dataset |
| 저장 | HF Dataset으로 저장 | 파인튜닝 준비 완료 |

### 3.2 분류 모델 아키텍처

**파일: `finetune_classifier.py`**

Llama-3.1-8B-Instruct 모델을 QLoRA(Quantized LoRA) 방식으로 파인튜닝합니다.

| 설정 항목 | 값 | 설명 |
|---------|---|------|
| **기본 모델** | Llama-3.1-8B-Instruct | Meta 공식 모델 |
| **양자화 방식** | 4-bit NF4 | 메모리 절약 |
| **LoRA Rank** | 16 | 속도 최적화 |
| **LoRA Alpha** | 16 | 가중치 스케일 |
| **LoRA Dropout** | 0.05 | 정규화 |
| **배치 사이즈** | 8 | Windows/NVIDIA 최적 |
| **학습률** | 2e-4 | AdamW Optimizer |
| **에포크** | 1 | 과적합 방지 |
| **옵티마이저** | adamw_torch | - |
| **GPU 메모리** | ~16GB | RTX 4090/A5000 권장 |

#### 양자화 설정 코드

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
```

#### LoRA 설정 코드

```python
peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"]
)
```

### 3.3 학습 프로세스 흐름

```
[Llama-3.1-8B-Instruct 모델 로드]
              ↓
        [4-bit 양자화]
              ↓
       [LoRA 어댑터 추가]
              ↓
     [합성 데이터셋 로드]
     (Train 800개 샘플)
              ↓
         [파인튜닝]
     (1 에포크, QLoRA)
              ↓
      [모델 체크포인트]
              ↓
   [테스트셋 평가]
   (Test 200개 샘플)
              ↓
   [분류기 모델 완성]
```

---

## 4. 프로젝트 구조

```
online_grooming_backsin/
│
├── generate_dataset.py              # 1,000개 합성 대화 생성 (Llama)
├── prepare_data_classifier.py       # 데이터 전처리 및 분할
├── finetune_classifier.py           # QLoRA 기반 분류기 학습
├── run_koalpaca.py                  # KoAlpaca 모델 테스트 (선택)
├── check_gpu.py                     # GPU 환경 확인
├── check.py                         # 기본 점검 스크립트
│
├── requirements.txt                 # 의존성 라이브러리
├── README.md                        # 프로젝트 설명
│
├── data/                            # 원본 데이터 폴더
│   └── PJZC.txt                    # 실제 대화 데이터 (필요시)
│
├── dataset_output_llama3_base/      # 생성된 합성 대화 (1,000개)
│   ├── scenario_1.txt
│   ├── scenario_2.txt
│   └── ...
│
├── final_data/                      # 전처리된 분류 데이터
│   └── grooming_classifier_dataset/
│       ├── train/
│       └── test/
│
├── fintuned_data/                   # 파인튜닝 데이터 (선택)
│
└── llama3.1-grooming-fast/          # 최종 학습된 모델
    ├── adapter_config.json
    └── adapter_model.bin
```

---

## 5. 설치 및 실행 가이드

### 5.1 환경 설정

```bash
# 레포지토리 클론
git clone https://github.com/hamret/online_grooming_backsin.git
cd online_grooming_backsin

# Python 3.10 이상 권장
python --version

# 의존성 설치
pip install -r requirements.txt

# PyTorch 설치 (CUDA 12.1 기준)
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 \
  --index-url https://download.pytorch.org/whl/cu121
```

### 5.2 합성 데이터 생성

```bash
python generate_dataset.py
```

**결과:**
- 폴더: `dataset_output_llama3_base/`
- 파일: 1,000개의 `scenario_*.txt` 파일
- 각 파일: 프롬프트 + 생성된 대화 내용

### 5.3 데이터 전처리

```bash
python prepare_data_classifier.py
```

**결과:**
- 폴더: `final_data/grooming_classifier_dataset/`
- 구성: Train (800개), Test (200개)
- 포맷: HuggingFace Datasets

### 5.4 분류기 학습

```bash
python finetune_classifier.py
```

**결과:**
- 폴더: `llama3.1-grooming-fast/`
- 파일: LoRA 어댑터 파일
- 학습 시간: GPU에 따라 1-3시간

### 5.5 GPU 확인 (선택)

```bash
python check_gpu.py
```

---

## 6. 필수 라이브러리

```
numpy<2
pandas
transformers==4.56.1
trl==0.25.1
peft==0.11.1
bitsandbytes==0.43.1
datasets>=2.18.0
accelerate>=0.31.0
tokenizers>=0.19.0
sentencepiece
safetensors
huggingface-hub
protobuf
einops
tqdm
wandb
```

**참고:** PyTorch는 시스템 CUDA 버전에 맞춰 별도 설치 필요합니다.

---

## 7. 실험 결과 및 평가

### 7.1 데이터셋 통계

| 항목 | 값 |
|------|---|
| 총 합성 샘플 | 1,000개 |
| Train 샘플 | 800개 |
| Test 샘플 | 200개 |
| 클래스 분포 | 1:1 (균형) |
| 평균 대화 길이 | 약 150-300 토큰 |
| 페르소나 조합 | 40 × 40 × 4 = 6,400 가능 조합 |

### 7.2 모델 학습 성능

| 항목 | 상태 | 설명 |
|------|------|------|
| 데이터 생성 | 완료 | 1,000개 샘플 성공 생성 |
| 데이터 전처리 | 완료 | 레이블링 및 분할 완료 |
| 모델 파인튜닝 | 완료 | QLoRA 학습 성공 |
| 정량 평가 | 진행 중 | 추가 성능 테스트 필요 |

---

## 8. 주요 한계점 및 개선 방향

### 8.1 현재 한계

| 항목 | 한계점 | 원인 |
|------|--------|------|
| 데이터 현실성 | 합성 대화의 표현 단조 | LLM의 패턴 한계 |
| 실제 성능 | 야생 데이터 검증 부재 | 실제 사례 데이터 없음 |
| 평가 기준 | 공개 벤치마크 없음 | 도메인 특화 데이터 부족 |
| 모델 비교 | 단일 모델만 테스트 | 시간/리소스 제약 |
| 일반화 능력 | 미검증 상태 | 도메인 적응 필요 |

### 8.2 향후 개선안

**단기 (1-2개월)**
- 프롬프트 엔지니어링 고도화
- 다양한 언어모델 비교 (GPT-4, Claude, Gemini)
- 평가 지표 확대 (Precision, Recall, F1-score)
- 자동화된 검증 스크립트 개발

**중기 (3-6개월)**
- 비식별 실제 데이터 수집 및 통합
- 공개 벤치마크 데이터셋 구축
- 도메인 특화 평가 프레임워크 개발
- 웹 API/서비스 배포

**장기 (6-12개월)**
- 다중 언어 모델 앙상블
- 멀티태스크 학습 (동시 분류 + 위험도 평가)
- 실시간 모니터링 시스템 구축
- 학술 논문 발표

---

## 9. 윤리 및 주의사항

### ⚠️ 중요 안내

1. **데이터 사용 목적**: 연구 및 교육 목적으로만 사용
2. **해로운 콘텐츠 생성 금지**: 실제 그루밍에 활용할 수 없음
3. **개인정보 보호**: 합성 데이터이므로 개인 식별 정보 없음
4. **모델 라이선스**: Llama는 Meta Llama Community License 준수
5. **재배포 제한**: 상업용 재배포 금지

---

## 10. 학술적 기여

본 프로젝트의 의의:

- **데이터 생성 패턴**: LLM 기반 도메인 특화 데이터 합성 방법론 제시
- **재현성**: 오픈소스로 제공되는 투명한 실험 설계
- **윤리적 접근**: 실제 데이터 없이도 연구 가능한 모델 제시
- **기술 최적화**: GPU 제한 환경에서의 QLoRA 파인튜닝 사례
- **사회적 기여**: 온라인 범죄 예방 AI 기초 연구 제공

---

## 11. 참고 자료

### 공식 문서
- [Llama 모델 가이드](https://www.llama.com/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/)
- [PEFT (LoRA) 문서](https://huggingface.co/docs/peft/)
- [bitsandbytes 문서](https://github.com/TimDettmers/bitsandbytes)

### 관련 연구
- QLoRA: Efficient Finetuning of Quantized LLMs
- LoRA: Low-Rank Adaptation of Large Language Models
- 온라인 그루밍 특성 및 예방 연구

---

## 12. 연락처 및 기여

### 이슈 및 피드백

이 프로젝트에 대한 제안, 버그 보고, 기여는 모두 환영합니다.

- **GitHub Issues**: 버그 보고 및 기능 요청
- **Pull Requests**: 코드 개선 및 패치
- **Discussions**: 학술 질문 및 토론

### 라이선스

MIT License - 자유로운 사용, 수정, 배포 가능

---

**프로젝트 상태**: ✅ 활발히 진행 중  
**마지막 업데이트**: 2025년 11월 18일  
**주요 기여자**: SKT Feelink Team
