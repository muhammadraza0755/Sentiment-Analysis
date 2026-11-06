"""RAG question answering over the comments.

question -> embed -> FAISS finds the most relevant comments -> an LLM answers
using ONLY those comments.
"""
import os
import numpy as np
import pandas as pd
import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"  # English + Urdu
PROMPT = """You are analysing the comment section of a clothing brand's YouTube ad.
Answer the question using ONLY the comments below. Comments may be in English,
Urdu or Roman Urdu. Summarise what people say (mention if opinions are mixed).
If the comments do not contain the answer, reply exactly: "Not mentioned in the comments."

Comments:
{context}

Question: {question}
Answer:"""


class CommentQA:
    def __init__(self, path="data/clean_comments.parquet"):
        self.df = pd.read_parquet(path)
        self.embedder = SentenceTransformer(EMBED_MODEL)
        emb_path = path.replace(".parquet", "_emb.npy")
        if os.path.exists(emb_path):
            emb = np.load(emb_path)
        else:
            emb = self.embedder.encode(self.df["text"].tolist(), batch_size=64,
                                       normalize_embeddings=True, show_progress_bar=True)
            np.save(emb_path, emb)
        self.index = faiss.IndexFlatIP(emb.shape[1])  # cosine similarity
        self.index.add(np.asarray(emb, dtype="float32"))
        self._local = None

    def retrieve(self, question, k=15):
        q = self.embedder.encode([question], normalize_embeddings=True).astype("float32")
        scores, idx = self.index.search(q, k)
        out = self.df.iloc[idx[0]].copy()
        out["similarity"] = scores[0]
        return out

    def answer(self, question, k=15):
        hits = self.retrieve(question, k)
        context = "\n".join(f"- {t[:300]}" for t in hits["text"])
        prompt = PROMPT.format(context=context, question=question)
        if os.getenv("ANTHROPIC_API_KEY"):
            return self._claude(prompt), hits
        return self._flan(prompt), hits

    def _claude(self, prompt):
        import anthropic
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5"),
            max_tokens=500, messages=[{"role": "user", "content": prompt}])
        return msg.content[0].text

    def _flan(self, prompt):
        # Free local fallback (English works best; weaker on Urdu).
        if self._local is None:
            from transformers import pipeline
            self._local = pipeline("text2text-generation", model="google/flan-t5-large")
        return self._local(prompt, max_new_tokens=150)[0]["generated_text"]


if __name__ == "__main__":
    qa = CommentQA()
    while True:
        q = input("\nAsk about the ad (or 'quit'): ").strip()
        if q.lower() in {"quit", "exit", ""}:
            break
        ans, _ = qa.answer(q)
        print(ans)
