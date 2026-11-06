# 👗 Ask the Comments: a Q&A model trained on a clothing ad's comment section

Pick a clothing-brand ad on YouTube (Khaadi, Gul Ahmed, Sapphire...) with 2000+ comments.
This project downloads the comments, cleans them, analyses them, and lets you **ask
questions about the product**. The answers come only from what viewers wrote.

## Architecture

```
YouTube Data API -> raw comments (Parquet) -> cleaning -> sentiment + topics
                                                   |
                          multilingual embeddings -> FAISS vector index
                                                   |
        question -> top-k similar comments -> LLM answers from them -> Streamlit UI
```

The model "learns" from the comments with **RAG (Retrieval-Augmented Generation)**:
comments are embedded into vectors, the most relevant ones are retrieved for each
question, and a language model writes the answer using only that evidence.
Training from scratch on ~2000 comments would give poor results; RAG is accurate,
explainable, and needs no GPU.

## Setup

```bash
git clone <your-repo-url> && cd comment-qa
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                              # then add your YOUTUBE_API_KEY
```

Get a free key: Google Cloud Console -> new project -> enable **YouTube Data API v3** -> Credentials -> API key.

## Run

```bash
python -m src.find_video          # lists ads with 2000+ comments; copy a video_id into .env
python -m src.collect             # downloads comments  -> data/raw_comments.parquet
python -m src.clean               # cleans them         -> data/clean_comments.parquet
python -m src.analyze             # sentiment + topics  -> data/analyzed_comments.parquet
streamlit run app.py              # chat UI + insights dashboard
```

Terminal-only version: `python -m src.qa`

## Answer model

- **Default (free, local):** `google/flan-t5-large`
- **Better quality:** set `ANTHROPIC_API_KEY` in `.env` and Claude is used automatically.

## Known limitations

- Many Pakistani comments are in **Roman Urdu**, which the sentiment model understands only partially.
- Keyword topic counts are a simple baseline; BERTopic would be a good extension.
- The YouTube API gives top-level comments plus a limited number of replies per thread.

## Ideas to extend (big-data angle)

- Collect comments from several ads/brands and compare them (Khaadi vs Gul Ahmed).
- Replace pandas cleaning with **PySpark** for multi-video datasets.
- Add BERTopic topic modelling and a word cloud.

## Tech

Python, pandas, Parquet, sentence-transformers, FAISS, Hugging Face Transformers, Streamlit.
