import pandas as pd
import numpy as np

INPUT_TRAIN = "train.csv"
INPUT_TEST = "test.csv"

OUTPUT_TRAIN = "train_clean.csv"
OUTPUT_TEST = "test_clean.csv"

def clean(df):
    # text를 전부 문자열 강제 변환
    df["text"] = df["text"].astype(str)

    # "nan", "None" 같은 문자열 제거
    df["text"] = df["text"].replace(["nan", "None", "none"], "", regex=True)

    # 너무 짧은 텍스트 제거
    df = df[df["text"].str.strip() != ""]

    # 라벨 오류 제거
    df = df[df["label"].astype(str).isin(["0", "1"])]

    # 형식 강제
    df["label"] = df["label"].astype(int)

    return df.reset_index(drop=True)


train_df = pd.read_csv(INPUT_TRAIN)
test_df = pd.read_csv(INPUT_TEST)

clean_train = clean(train_df)
clean_test = clean(test_df)

clean_train.to_csv(OUTPUT_TRAIN, index=False)
clean_test.to_csv(OUTPUT_TEST, index=False)

print("정상 저장 완료:")
print(" →", OUTPUT_TRAIN, len(clean_train))
print(" →", OUTPUT_TEST, len(clean_test))
