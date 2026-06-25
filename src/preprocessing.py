# set of allowed characters
allowed = set([chr(i) for i in range(97, 123)] + [str(i) for i in range(10)] + [" "])


def clean(text: str) -> str:
    text = text.lower()
    text = "".join(char for char in text if char in allowed)    # filtering characters
    text = " ".join(text.split())   # space normalization
    return text
