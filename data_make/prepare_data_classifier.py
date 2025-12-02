import json
import random
from datasets import Dataset

# 원본 데이터 파일 (다운로드한 파일)
INPUT_FILE = r"C:\Users\user\Desktop\KoAlpaca-Polyglot-5.8B\data\PJZC.txt"
# 파인튜닝에 사용할 학습 데이터 파일 (결과물)
OUTPUT_DATASET_NAME = r"C:\Users\user\Desktop\KoAlpaca-Polyglot-5.8B\final_data\grooming_classifier_dataset"


# 대화 포맷팅 함수 (이전과 약간 다름)
def format_conversation(messages):
    if not messages:
        return None

    groomer_id = messages[0]['author']
    convo_text = []
    for msg in messages:
        author = msg.get('author', 'unknown')
        text = msg.get('text', '').strip()

        if not text:
            continue

        if author == groomer_id:
            convo_text.append(f"User A: {text}")
        else:
            convo_text.append(f"User B: {text}")

    # 너무 짧은 대화는 학습에서 제외
    if len(convo_text) < 4:
        return None

    return "\n".join(convo_text)


def create_classifier_dataset():
    print(f"Loading {INPUT_FILE}...")
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"[오류] {INPUT_FILE}을 찾을 수 없습니다. N_G_data.txt 파일이 이 스크립트와 같은 폴더에 있는지 확인하세요.")
        return
    except Exception as e:
        print(f"[오류] 파일 로딩 중 에러: {e}")
        return

    conversations = data.get('conversation', [])
    print(f"Found {len(conversations)} total conversations.")

    processed_data = []
    for convo in conversations:
        label = convo.get('label')
        # 레이블이 0 또는 1이 아니면 건너뜁니다.
        if label not in ['0', '1']:
            continue

        formatted_chat = format_conversation(convo.get('messages', []))

        if formatted_chat:
            # Llama 3 Instruct 모델에게 "분류" 작업을 지시하는 프롬프트를 만듭니다.
            # "text" 필드와 "label" 필드로 구성합니다.

            # Llama 3의 공식 프롬프트 형식을 사용합니다.
            prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nAnalyze the following conversation and classify it as 'Grooming' (1) or 'Normal' (0). Provide only the label (0 or 1) as your answer.\n\n<CONVERSATION>\n{formatted_chat}\n</CONVERSATION><|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{label}<|eot_id|>"

            processed_data.append({
                "text": prompt,  # 학습할 전체 텍스트
                "label": int(label)  # 분류용 레이블 (숫자 0 또는 1)
            })

    print(f"Successfully processed {len(processed_data)} conversations for classification.")

    if not processed_data:
        print("[오류] 처리할 데이터가 없습니다. N_G_data.txt 파일 내용을 확인하세요.")
        return

    # Hugging Face Dataset 객체로 변환
    hf_dataset = Dataset.from_list(processed_data)

    # 80%는 학습(train), 20%는 평가(test)용으로 분리
    hf_dataset = hf_dataset.train_test_split(test_size=0.2)

    # 디스크에 저장 (나중에 finetune.py가 이 폴더를 읽음)
    hf_dataset.save_to_disk(OUTPUT_DATASET_NAME)

    print(f"Successfully saved classifier dataset to folder: {OUTPUT_DATASET_NAME}")
    print("\nDataset structure:")
    print(hf_dataset)
    print("\nExample (train):")
    print(hf_dataset['train'][0]['text'])


if __name__ == "__main__":
    create_classifier_dataset()