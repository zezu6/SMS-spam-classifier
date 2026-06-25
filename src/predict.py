from sys import argv
from preprocessing import clean
from joblib import load

pipe = load("../model/spam_classifier.joblib")

try:
    sms = clean(argv[1])
except IndexError:
    print('usage: python predict.py "your sms text"')

pred = pipe.predict([sms])
if pred[0] == "ham":
    conf = pipe.predict_proba([sms])[0][0]
else:
    conf = pipe.predict_proba([sms])[0][1]
conf = round(conf, 2)

print(f"prediction: {pred[0]}\nconfidence: {conf}")
