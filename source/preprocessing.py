import pandas as pd

df = pd.read_csv(filepath_or_buffer="../data/spam.csv", encoding="latin-1")
df = df[["v1", "v2"]]
df.rename(columns={"v1": "output", "v2": "input"}, inplace=True)

# set of allowed characters
allowed = set([chr(i) for i in range(97, 123)] + [str(i) for i in range(10)] + [" "])


def clean(text: str) -> str:
    text = text.lower()
    text = "".join(char for char in text if char in allowed)    # filtering characters
    text = " ".join(text.split())   # space normalization
    return text


df["input"] = df["input"].apply(clean)
print(df.head())
