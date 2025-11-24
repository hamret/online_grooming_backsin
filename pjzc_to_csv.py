import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

PJZC_FILE = r"C:\Users\user\PycharmProjects\online_grooming_backsin\data\PJZC.txt"
PJZ_FILE = r"C:\Users\user\PycharmProjects\online_grooming_backsin\data\PJZ.txt "

OUT_TRAIN = "train.csv"
OUT_TEST = "test.csv"

def load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for convo in data["conversation"]:
        label = int(convo["label"])

        for m in convo["messages"]:
            text = m["text"].strip()
            if not text:
                continue

            rows.append({
                "text": text,
                "label": label
            })

    return pd.DataFrame(rows)


print("📥 Loading PJZC...")
pjzc_df = load_dataset(PJZC_FILE)

print("📥 Loading PJZ...")
pjz_df = load_dataset(PJZ_FILE)

df = pd.concat([pjzc_df, pjz_df], ignore_index=True)

print("전체 메시지 수:", len(df))
print(df["label"].value_counts())

label0 = df[df["label"] == 0]
label1 = df[df["label"] == 1]

print("\n⚠ BEFORE Oversampling:")
print(df["label"].value_counts())

label1_over = resample(
    label1,
    replace=True,
    n_samples=len(label0),
    random_state=42
)

df_balanced = pd.concat([label0, label1_over], ignore_index=True)
df_balanced = df_balanced.sample(frac=1, random_state=42)

print("\n✅ AFTER Oversampling:")
print(df_balanced["label"].value_counts())

train_df, test_df = train_test_split(
    df_balanced,
    test_size=0.2,
    random_state=42,
    stratify=df_balanced["label"]
)

train_df.to_csv(OUT_TRAIN, index=False, encoding="utf-8")
test_df.to_csv(OUT_TEST, index=False, encoding="utf-8")

print("\n🎉 SAVED:", OUT_TRAIN, OUT_TEST)
