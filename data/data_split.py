import pandas as pd
from sklearn.model_selection import train_test_split
import json


mbpp_path = "./mbpp_text_code_augmented.jsonl"
full_augmented_path = "./cleaned_augmented_final_data.jsonl"
mbpp_df = pd.read_json(mbpp_path, lines=True)
augmented_df = pd.read_json(full_augmented_path, lines=True)


mbpp_df = mbpp_df.rename(columns={"original_text": "text", "original_code": "code"})

# 20% MBPP for test
mbpp_train, mbpp_test = train_test_split(mbpp_df, test_size=0.2, random_state=42)

# 80 % MBPP +  100% Augmented for train

train_df = pd.concat([augmented_df, mbpp_train], ignore_index=True)
test_df = mbpp_test
train_df = train_df.drop_duplicates(subset=["text", "code"])

# test_keys = set(zip(mbpp_test["text"], mbpp_test["code"]))
# train_df = train_df[~train_df.apply(lambda row: (row["text"], row["code"]) in test_keys, axis=1)]


# OUTPUT
train_df.to_json("./train_final.jsonl", orient="records", lines=True)
test_df.to_json("./test_final.jsonl", orient="records", lines=True)

print(f"Train size: {len(train_df)}")
print(f"Test size: {len(test_df)}")
