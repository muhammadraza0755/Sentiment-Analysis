"""Sentiment + topic analysis of the cleaned comments."""
import pandas as pd
from transformers import pipeline

# Multilingual sentiment model (works on English and Urdu script;
# Roman Urdu is only partially understood - mention this in your report).
MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

TOPICS = {
    "price": ["price", "expensive", "costly", "cheap", "mehnga", "mehngi", "qeemat", "rate", "afford", "قیمت", "مہنگا"],
    "quality / fabric": ["quality", "fabric", "cloth", "kapra", "kapray", "stitch", "material", "cotton", "lawn", "کپڑا"],
    "design / style": ["design", "color", "colour", "print", "style", "pattern", "beautiful", "stunning", "khoobsurat", "ڈیزائن"],
    "model / ad": ["model", "actress", "song", "music", "ad ", "advert", "commercial", "video", "ماڈل"],
    "sale / delivery": ["sale", "discount", "delivery", "shipping", "order", "store", "outlet", "online", "ڈسکاؤنٹ"],
    "size / fit": ["size", "fit", "length", "sizes"],
}


def add_sentiment(df, batch_size=32):
    clf = pipeline("sentiment-analysis", model=MODEL, truncation=True,
                   max_length=128, batch_size=batch_size)
    preds = clf(df["text"].tolist())
    df = df.copy()
    df["sentiment"] = [p["label"].lower() for p in preds]
    df["sentiment_score"] = [round(p["score"], 3) for p in preds]
    return df


def topic_counts(df):
    low = df["text"].str.lower()
    return pd.Series({t: int(low.apply(lambda x: any(k in x for k in kws)).sum())
                      for t, kws in TOPICS.items()}).sort_values(ascending=False)


def run(src="data/clean_comments.parquet", dst="data/analyzed_comments.parquet"):
    df = add_sentiment(pd.read_parquet(src))
    df.to_parquet(dst, index=False)
    print(df["sentiment"].value_counts(normalize=True).round(3).to_string())
    print(topic_counts(df).to_string())
    print(f"Saved {dst}")


if __name__ == "__main__":
    run()
