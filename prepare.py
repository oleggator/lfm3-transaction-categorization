"""Sample a balanced train set and a held-out eval set (disjoint descriptions) as chat messages."""
import json

import pandas as pd

TRAIN_PER_CLASS = 1000
EVAL_PER_CLASS = 200
SEED = 0

df = pd.read_parquet("data/train.parquet", columns=["transaction_description", "category"])
df = df.drop_duplicates()
# descriptions labelled with more than one category are unlearnable noise
df = df[~df.transaction_description.duplicated(keep=False)]
categories = sorted(df.category.unique())

SYSTEM = (
    "Classify the bank transaction into exactly one category. Reply with the category name only.\n"
    "Categories: " + "; ".join(categories)
)


def to_messages(row):
    return {"messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": row.transaction_description},
        {"role": "assistant", "content": row.category},
    ]}


sample = df.groupby("category").sample(TRAIN_PER_CLASS + EVAL_PER_CLASS, random_state=SEED)
eval_df = sample.groupby("category").head(EVAL_PER_CLASS).sample(frac=1, random_state=SEED)
train_df = sample.drop(eval_df.index).sample(frac=1, random_state=SEED)
assert not set(train_df.transaction_description) & set(eval_df.transaction_description)

for name, part in [("train", train_df), ("eval", eval_df)]:
    with open(f"data/{name}.jsonl", "w") as f:
        for row in part.itertuples():
            f.write(json.dumps(to_messages(row)) + "\n")
    print(name, len(part))
