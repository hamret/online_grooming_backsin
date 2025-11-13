import json
import random

# 원본 데이터 파일 (다운로드한 파일)
INPUT_FILE = r"C:\Users\user\Desktop\KoAlpaca-Polyglot-5.8B\data\PJZ.txt"
# 파인튜닝에 사용할 학습 데이터 파일 (결과물)
OUTPUT_FILE = r"C:\Users\user\Desktop\KoAlpaca-Polyglot-5.8B\fintuned_data\grooming_train_data.txt"
# 학습에 사용할 최대 대화 수 (너무 많으면 4070에서 오래 걸림)
MAX_SAMPLES = 5000


def format_conversation(messages):
    """
    대화 메시지 리스트를 'User A: ... User B: ...' 형식의
    단일 텍스트 덩어리로 변환합니다.
    """
    full_conversation = ""

    # PJZ 데이터셋은 author 이름이 'decoy', 'Billy Joe' 등 제각각입니다.
    # 우리는 이것을 일관되게 'User A' (가해자)와 'User B' (피해자)로 바꿔야 합니다.

    if not messages:
        return None

    # 첫 번째 메시지를 보낸 사람을 'User A' (가해자)로 가정합니다.
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

    # 10턴(turn) 미만의 너무 짧은 대화는 학습에서 제외
    if len(convo_text) < 10:
        return None

    return "\n".join(convo_text)


def create_training_file():
    print(f"Loading {INPUT_FILE}...")
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"[오류] {INPUT_FILE}이 완전한 JSON 형식이 아닙니다.")
        print("파일 상단 '{\"conversation\": [' 로 시작하고, 맨 끝이 ']}'로 끝나는지 확인하세요.")
        print("PJZ.txt 파일이 손상되었을 수 있습니다.")
        return
    except FileNotFoundError:
        print(f"[오류] {INPUT_FILE}을 찾을 수 없습니다. PJZ.txt 파일이 이 스크립트와 같은 폴더에 있는지 확인하세요.")
        return

    conversations = data.get('conversation', [])
    if not conversations:
        print("[오류] JSON 파일에서 'conversation' 키를 찾을 수 없습니다.")
        return

    print(f"Found {len(conversations)} total conversations.")

    grooming_chats = []
    for convo in conversations:
        # "label": "1" 인 (그루밍 대화)만 필터링합니다.
        if convo.get('label') == '1':
            formatted_chat = format_conversation(convo.get('messages', []))
            if formatted_chat:
                # Base 모델 학습을 위해 "Few-Shot" 프롬프트 형식으로 최종 가공
                # <|begin_of_text|>는 Llama 3의 문장 시작 토큰입니다.
                final_prompt = f"<|begin_of_text|>\nA chat between a groomer (User A) and a victim (User B).\n\n{formatted_chat}\n<|eot_id|>"
                grooming_chats.append(final_prompt)

    print(f"Found and formatted {len(grooming_chats)} grooming conversations (min. 10 turns).")

    # 샘플링 (너무 많으면 학습이 오래 걸리므로)
    if len(grooming_chats) > MAX_SAMPLES:
        print(f"Sampling down to {MAX_SAMPLES} conversations...")
        grooming_chats = random.sample(grooming_chats, MAX_SAMPLES)

    # 최종 학습 파일을 저장합니다.
    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            # 각 대화는 \n\n\n (줄바꿈 3번)으로 구분합니다.
            f.write("\n\n\n".join(grooming_chats))
        print(f"Successfully created training file: {OUTPUT_FILE}")
    except Exception as e:
        print(f"Error writing file: {e}")


if __name__ == "__main__":
    create_training_file()