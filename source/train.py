import pandas as pd
from preprocessing import clean
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


df = pd.read_csv(filepath_or_buffer="../data/spam.csv", encoding="latin-1")
df = df[["v1", "v2"]]
df.rename(columns={"v1": "output", "v2": "input"}, inplace=True)

df["input"] = df["input"].apply(clean)

X = df["input"]
y = df["output"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=10, stratify=y
)

vectorizer = TfidfVectorizer(stop_words="english", min_df=2)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X=X_train_vec, y=y_train)

y_pred = model.predict(X=X_test_vec)

accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
precision = precision_score(y_true=y_test, y_pred=y_pred, average=None)
recall = recall_score(y_true=y_test, y_pred=y_pred, average=None)
f1 = f1_score(y_true=y_test, y_pred=y_pred, average=None)
conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred, )

print(f"accuracy: {accuracy}\nprecision: {precision}\nrecall: {recall}\nf1: {f1}")
print(f"confusion matrix: {conf_matrix}")
