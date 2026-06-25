# SMS Spam Classifier

Simple SMS spam classifier built with TF-IDF and Naive Bayes.

The model predicts whether a message is spam or ham and returns a confidence score.

This is my first machine learning project, created purely for learning purposes.

## Example

```bash
cd src
python predict.py "Knicks are now champions of the NBA!"
```

Output:

```text
prediction: ham
confidence: 0.87
```

## Requirements

* pandas
* scikit-learn
* joblib

## Usage

```bash
cd src
python predict.py "your message"
```

## Note

Run `predict.py` from the `src/` directory.

## Dataset

Kaggle SMS spam dataset.
