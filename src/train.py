import pandas as pd
from preprocessing import clean
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from joblib import dump

df = pd.read_csv(filepath_or_buffer="../data/spam.csv", encoding="latin-1")
df = df[["v1", "v2"]]
df.rename(columns={"v1": "output", "v2": "input"}, inplace=True)

df["input"] = df["input"].apply(clean)
X = df["input"]
y = df["output"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=10, stratify=y
)

pipe = Pipeline(steps=[
    ("vectorizer", TfidfVectorizer(stop_words="english", min_df=2)),
    ("classifier", MultinomialNB())
])

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

report = classification_report(y_true=y_test, y_pred=y_pred)
conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)

print(f"report:\n{report}")
print(f"confusion matrix:\n{conf_matrix}")

dump(pipe, "../model/spam_classifier.joblib")
