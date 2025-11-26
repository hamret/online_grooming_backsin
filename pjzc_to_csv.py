# prepare_sentence_dataset.py
import json
import pandas as pd
import nltk
from sklearn.utils import resample

nltk.download("punkt")
nltk.download("punkt_tab")

PJZ_PATH = r"C:\Users\user\PycharmProjects\online_grooming_backsin\data\PJZ.txt"      # optional
PJZC_PATH = r"C:\Users\user\PycharmProjects\online_grooming_backsin\data\PJZC.txt"    # required

OUT_TRAIN = "train.csv"
OUT_TEST = "test.csv"

def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for convo in data["conversation"]:
        label = int(convo["label"])
        for msg in convo["messages"]:
            text = msg["text"].strip()
            if not text:
                continue
            # 문장 단위 split
            sentences = nltk.sent_tokenize(text)
            for s in sentences:
                rows.append({"text": s, "label": label})
    return pd.DataFrame(rows)

print("\n--- Loading datasets ---")
df_list = []

# PJZC는 반드시 존재
df_list.append(load_data(PJZC_PATH))

# PJZ는 선택적(있으면 추가)
try:
    df_list.append(load_data(PJZ_PATH))
except:
    print("PJZ not found. Only using PJZC.")

df = pd.concat(df_list, ignore_index=True)
print("Total sentences:", len(df))

# 🔥 라벨 1(그루밍) oversampling
grooming = df[df["label"] == 1]
normal = df[df["label"] == 0]

grooming_up = resample(
    grooming,
    replace=True,
    n_samples=len(normal),   # 1:1 맞추기
    random_state=42
)

df_balanced = pd.concat([normal, grooming_up]).sample(frac=1, random_state=42)

# Train/Test split
from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(
    df_balanced,
    test_size=0.2,
    random_state=42,
    stratify=df_balanced["label"]
)

train_df.to_csv(OUT_TRAIN, index=False)
test_df.to_csv(OUT_TEST, index=False)

print("Saved:", OUT_TRAIN, OUT_TEST)
