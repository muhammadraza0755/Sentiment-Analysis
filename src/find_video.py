"""Find clothing-brand ads on YouTube that have 2000+ comments."""
import os
from dotenv import load_dotenv
from googleapiclient.discovery import build

load_dotenv()
yt = build("youtube", "v3", developerKey=os.getenv("YOUTUBE_API_KEY"))

QUERIES = [
    "Khaadi ad", "Khaadi campaign", "Gul Ahmed ad", "Gul Ahmed lawn campaign",
    "Sapphire ad", "Alkaram ad", "Nishat Linen ad", "Sana Safinaz ad",
]
MIN_COMMENTS = 2000


def search_ids(query, n=25):
    res = yt.search().list(q=query, part="id", type="video",
                           maxResults=n, order="viewCount").execute()
    return [i["id"]["videoId"] for i in res["items"]]


def get_stats(ids):
    out = []
    for i in range(0, len(ids), 50):
        res = yt.videos().list(part="snippet,statistics",
                               id=",".join(ids[i:i + 50])).execute()
        for v in res["items"]:
            s = v["statistics"]
            out.append({
                "video_id": v["id"],
                "title": v["snippet"]["title"],
                "channel": v["snippet"]["channelTitle"],
                "views": int(s.get("viewCount", 0)),
                "comments": int(s.get("commentCount", 0)),
            })
    return out


if __name__ == "__main__":
    ids = list({vid for q in QUERIES for vid in search_ids(q)})
    rows = [r for r in get_stats(ids) if r["comments"] >= MIN_COMMENTS]
    rows.sort(key=lambda r: r["comments"], reverse=True)
    if not rows:
        print("No video with enough comments found. Add more queries in QUERIES.")
    for r in rows:
        print(f'{r["video_id"]} | {r["comments"]:>6} comments | '
              f'{r["views"]:>9} views | {r["channel"]} | {r["title"][:70]}')
    print("\nCopy one video_id into VIDEO_ID in your .env file.")
