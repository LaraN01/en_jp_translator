import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Sentiment Analysis", page_icon="😊", layout="centered")

st.title("😊 Sentiment Analysis")
st.caption("Type text and get a sentiment prediction (POSITIVE / NEGATIVE).")

@st.cache_resource
def load_sentiment_pipeline():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=-1  
    )

sentiment = load_sentiment_pipeline()

text = st.text_area("Enter text", height=160, placeholder="I really enjoyed this product...")

col1, col2 = st.columns([1, 1])
analyze = col1.button("Analyze", type="primary", use_container_width=True)
clear = col2.button("Clear", use_container_width=True)

if clear:
    st.session_state["result"] = None
    st.rerun()

if analyze:
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Analyzing..."):
            try:
                result = sentiment(text[:2000])[0]  
                st.session_state["result"] = result
            except Exception as e:
                st.error(f"Error: {e}")

res = st.session_state.get("result")

if res:
    label = res["label"]
    score = float(res["score"])

    st.subheader("Result")
    st.metric("Sentiment", label, f"{score*100:.1f}% confidence")

    if label.upper() == "POSITIVE":
        st.success("Overall sentiment looks positive.")
    else:
        st.error("Overall sentiment looks negative.")

    with st.expander("Raw output"):
        st.json(res)
