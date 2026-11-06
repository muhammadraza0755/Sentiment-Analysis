import json
import pandas as pd

with open("data/video.info.json", encoding="utf-8") as f:
    info = json.load(f)

rows = [{
    "comment_id": c["id"],
    "text": c["text"],
    "likes": c.get("like_count") or 0,
    "date": c.get("timestamp"),
    "is_reply": c.get("parent", "root") != "root",
} for c in info.get("comments", [])]

df = pd.DataFrame(rows)
df.to_parquet("data/raw_comments.parquet", index=False)
df.head(200).to_csv("data/sample_comments.csv", index=False)
print(f"Saved {len(df)} comments")