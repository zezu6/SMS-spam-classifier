import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

from preprocessing import clean

df = pd.read_csv(filepath_or_buffer="../data/spam.csv", encoding="latin-1")
df = df[["v1", "v2"]]
df.rename(columns={"v1": "output", "v2": "input"}, inplace=True)

df["input"] = df["input"].apply(clean)

X = df["input"]
y = df["output"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=10, stratify=y
)

print(type(X_train))
