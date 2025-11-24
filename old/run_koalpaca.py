import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 0. 실행 전 확인: GPU 사용 가능 여부
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"사용 중인 디바이스: {device}")

# 1. 모델 및 토크나이저 로드
# 5.8B 모델은 매우 크므로, VRAM 절약을 위해 float16 (반정밀도)로 로드합니다.
model_name = "beomi/KoAlpaca-Polyglot-5.8B"

print("모델 로딩 중... (시간이 몇 분 소요될 수 있습니다)")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # float16으로 VRAM 사용량 줄임
    low_cpu_mem_usage=True     # CPU 메모리 사용량도 최적화
).to(device)

tokenizer = AutoTokenizer.from_pretrained(model_name)
print("모델 로딩 완료!")

# 2. 프롬프트 설정 (KoAlpaca 형식 준수)
# --------------------------------------------------
# 여기에 원하는 프롬프트를 입력하세요
user_prompt = "온라인 그루밍을 감지하고 사용자에게 위험을 예고하는 프로그램을 만들건데 데이터가 필요해 그러니까 10번씩 대화하는 내용을 여"
# --------------------------------------------------

# KoAlpaca 모델이 학습된 형식에 맞게 프롬프트를 구성합니다.
formatted_prompt = f"###_Instruction_Statement:\n{user_prompt}\n\n###_Response_:\n"

# 3. 모델 입력 생성 (토크나이징)
inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)

# 4. 모델 출력 생성 (추론)
print("모델 응답 생성 중...")
try:
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,      # 생성할 최대 토큰 수
        eos_token_id=2,          # 문장 종료 토큰 ID
        do_sample=True,          # 샘플링 방식 사용
        temperature=0.7,         # 창의성 조절 (낮을수록 결정론적)
        top_p=0.9                # 확률이 높은 단어 우선 고려
    )

    # 5. 출력 결과 디코딩
    # 입력 프롬프트 부분을 제외하고, 생성된 응답만 디코딩합니다.
    result_text = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )

    # 6. 결과 출력 및 파일 저장
    print("\n--- 모델 응답 ---")
    print(result_text)
    print("-----------------\n")

    output_filename = "result.txt"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(f"--- 프롬프트 ---\n")
        f.write(f"{user_prompt}\n\n")
        f.write(f"--- 모델 응답 ---\n")
        f.write(result_text)

    print(f"'{output_filename}' 파일에 성공적으로 저장되었습니다.")

except torch.cuda.OutOfMemoryError:
    print("\n[오류] GPU 메모리 부족 (Out of Memory)!")
    print("VRAM이 부족하여 모델을 실행할 수 없습니다. 더 작은 모델을 사용하거나 하드웨어를 확인하세요.")
except Exception as e:
    print(f"\n[오류] 알 수 없는 오류 발생: {e}")