import os
import pandas as pd
import streamlit as st
from src.qa import CommentQA
from src.analyze import topic_counts

st.set_page_config(page_title="Ask the Comments", page_icon="👗", layout="wide")
st.title("👗 Ask the Comments")
st.caption("A RAG model that answers questions using only the YouTube comments of a clothing ad.")

if not os.path.exists("data/clean_comments.parquet"):
    st.error("Run the pipeline first: collect -> clean (see README).")
    st.stop()


@st.cache_resource
def load_qa():
    return CommentQA()


@st.cache_data
def load_df():
    p = "data/analyzed_comments.parquet"
    return pd.read_parquet(p if os.path.exists(p) else "data/clean_comments.parquet")


qa, df = load_qa(), load_df()
tab_ask, tab_insights = st.tabs(["💬 Ask", "📊 Insights"])

with tab_ask:
    st.write(f"Model is grounded in **{len(df):,}** comments.")
    examples = ["What do people say about the price?", "Do people like the fabric quality?",
                "What do people dislike about this ad?", "Which design or colour do people love?"]
    cols = st.columns(len(examples))
    for c, e in zip(cols, examples):
        if c.button(e):
            st.session_state["q"] = e
    q = st.text_input("Your question", key="q")
    if q:
        with st.spinner("Reading the comments..."):
            ans, hits = qa.answer(q)
        st.subheader("Answer")
        st.write(ans)
        with st.expander("Comments used as evidence"):
            st.dataframe(hits[["text", "likes", "similarity"]], use_container_width=True)

with tab_insights:
    c1, c2 = st.columns(2)
    c1.metric("Total comments", f"{len(df):,}")
    c2.metric("Total likes", f"{int(df['likes'].sum()):,}")
    if "sentiment" in df:
        st.subheader("Sentiment")
        st.bar_chart(df["sentiment"].value_counts())
    else:
        st.info("Run `python -m src.analyze` to see sentiment.")
    st.subheader("What people talk about")
    st.bar_chart(topic_counts(df))
    st.subheader("Most liked comments")
    st.dataframe(df.nlargest(15, "likes")[["text", "likes"]], use_container_width=True)
