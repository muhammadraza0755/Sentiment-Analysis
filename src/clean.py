"""Clean raw comments: strip html/links, drop duplicates, spam and tiny comments."""
import html
import re
import pandas as pd

URL = re.compile(r"https?://\S+|www\.\S+")
TAG = re.compile(r"<[^>]+>")
URDU = re.compile(r"[\u0600-\u06FF]")


def clean_text(t):
    t = html.unescape(str(t))
    t = TAG.sub(" ", t)
    t = URL.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


def is_spam(t):
    if re.search(r"(.)\1{7,}", t):          # aaaaaaaa
        return True
    words = t.lower().split()
    return len(words) > 5 and len(set(words)) == 1


def run(src="data/raw_comments.parquet", dst="data/clean_comments.parquet"):
    df = pd.read_parquet(src)
    n0 = len(df)
    df["text"] = df["text"].map(clean_text)
    df = df[df["text"].str.len() >= 3]
    df = df[~df["text"].map(is_spam)]
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    df["script"] = df["text"].map(lambda t: "urdu" if URDU.search(t) else "latin")
    df.to_parquet(dst, index=False)
    print(f"{n0} -> {len(df)} comments after cleaning. Saved {dst}")
    print(df["script"].value_counts().to_string())


if __name__ == "__main__":
    run()
