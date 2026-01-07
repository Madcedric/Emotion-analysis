from datasets import load_dataset
import pandas as pd

dataset = load_dataset("SetFit/emotion")

# This dataset has ONLY train split
df = pd.DataFrame(dataset['train'])

print(df.shape)
print(df.columns)

df.to_csv("emotion_data_1.csv", index=False)
print("Dataset saved successfully")
