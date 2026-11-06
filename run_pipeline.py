import json, os, subprocess, sys
import pandas as pd
from src import clean

URL = sys.argv[1] if len(sys.argv) > 1 else "https://www.youtube.com/watch?v=8kLu8wz2ehA"
os.makedirs("data", exist_ok=True)

print("1/3 Downloading comments (can take a few minutes)...")
r = subprocess.run([sys.executable, "-m", "yt_dlp", "--skip-download", "--write-comments",
                    "--extractor-args", "youtube:max_comments=5000",
                    "-o", "data/video", URL])
if r.returncode != 0 or not os.path.exists("data/video.info.json"):
    sys.exit("FAILED at step 1: yt-dlp did not create data/video.info.json. Run: pip install -U yt-dlp")

print("2/3 Converting to table...")
with open("data/video.info.json", encoding="utf-8") as f:
    info = json.load(f)
rows = [{"comment_id": c["id"], "text": c["text"], "likes": c.get("like_count") or 0,
         "date": c.get("timestamp"), "is_reply": c.get("parent", "root") != "root"}
        for c in info.get("comments", [])]
if not rows:
    sys.exit("FAILED at step 2: the video has no downloaded comments.")
df = pd.DataFrame(rows)
df.to_parquet("data/raw_comments.parquet", index=False)
df.head(200).to_csv("data/sample_comments.csv", index=False)
print(f"Saved {len(df)} comments")

print("3/3 Cleaning...")
clean.run()
print("\nDONE. Now run: python -m streamlit run app.py")
