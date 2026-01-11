import streamlit as st
import torch
from transformers import (
    pipeline,
    M2M100ForConditionalGeneration,
    M2M100Tokenizer,
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from langdetect import detect

# -----------------------------
# Streamlit Config
# -----------------------------
st.set_page_config(
    page_title="News Categorization & Summarization",
    layout="wide"
)

st.title("📰 News Categorization & Summarization")
st.write(
    "Paste a news article in any language. "
    "The app detects the language, translates it to English if needed, "
    "categorizes the news, and generates an English summary."
)

# -----------------------------
# Load Models (cached)
# -----------------------------
@st.cache_resource
def load_models():
    # Summarization model
    summarizer = pipeline(
        "summarization",
        model="facebook/bart-large-cnn"
    )

    # Translation model
    translator_tokenizer = M2M100Tokenizer.from_pretrained(
        "facebook/m2m100_418M"
    )
    translator_model = M2M100ForConditionalGeneration.from_pretrained(
        "facebook/m2m100_418M"
    )

    # News categorization model
    clf_tokenizer = AutoTokenizer.from_pretrained("news_classifier")
    clf_model = AutoModelForSequenceClassification.from_pretrained(
        "news_classifier"
    )
    clf_model.eval()

    return (
        summarizer,
        translator_tokenizer,
        translator_model,
        clf_tokenizer,
        clf_model
    )


(
    summarizer,
    translator_tokenizer,
    translator_model,
    clf_tokenizer,
    clf_model
) = load_models()

labels = ["World", "Sports", "Business", "Sci/Tech"]

# -----------------------------
# Helper Functions (UNCHANGED)
# -----------------------------
def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"


def translate(text, src, tgt):
    translator_tokenizer.src_lang = src
    encoded = translator_tokenizer(text, return_tensors="pt", truncation=True)

    generated = translator_model.generate(
        **encoded,
        forced_bos_token_id=translator_tokenizer.get_lang_id(tgt)
    )

    return translator_tokenizer.decode(
        generated[0],
        skip_special_tokens=True
    )


def summarize_text(text):
    return summarizer(
        text,
        max_length=130,
        min_length=40,
        do_sample=False
    )[0]["summary_text"]


def categorize_news(text):
    inputs = clf_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256
    )

    with torch.no_grad():
        outputs = clf_model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)
    confidence, predicted_class = torch.max(probs, dim=1)

    return {
        "category": labels[predicted_class.item()],
        "confidence": round(confidence.item(), 3)
    }

# -----------------------------
# Streamlit UI
# -----------------------------
article = st.text_area(
    "Paste your article here:",
    height=300
)

translate_to_english = st.checkbox("Translate summary to English")

if st.button("Analyze"):
    if article.strip() == "":
        st.warning("Please paste an article.")
    else:
        with st.spinner("Processing..."):
            lang = detect_language(article)

            processed_text = article
            if lang != "en":
                processed_text = translate(article, lang, "en")

            category_result = categorize_news(processed_text)
            summary = summarize_text(processed_text)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🔍 Analysis")
            st.write(f"**Detected Language:** {lang}")
            st.write(f"**Category:** {category_result['category']}")
            st.write(f"**Confidence:**
