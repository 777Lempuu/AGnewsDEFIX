import streamlit as st
import torch
import torch.nn as nn
from transformers import AutoTokenizer

# ------------------- Model Definition -------------------

class SimpleNewsClassifier(nn.Module):
    def __init__(self, vocab_size=30522, embed_dim=128, num_classes=4):
        super(SimpleNewsClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, input_ids):
        embeds = self.embedding(input_ids)
        pooled = embeds.mean(dim=1)  # simple average pooling
        return self.fc(pooled)

# ------------------- Constants -------------------

CLASS_LABELS = {
    0: "World 🌍",
    1: "Sports ⚽",
    2: "Business 💼",
    3: "Sci/Tech 🔬"
}

# ------------------- Streamlit Setup -------------------

st.set_page_config(page_title="AG News - DeFix Model", layout="wide")
st.title("📰 AG News Classifier (DeFix Model)")

@st.cache_resource
def load_model_and_tokenizer():
    try:
        model = SimpleNewsClassifier()
        state_dict = torch.load("AG_DeFix.pt", map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained("google/bert_uncased_L-2_H-128_A-2")
        st.success("✅ Model and tokenizer loaded!")
        return model, tokenizer
    except Exception as e:
        st.error(f"❌ Failed to load model/tokenizer: {e}")
        return None, None

model, tokenizer = load_model_and_tokenizer()

# ------------------- Prediction Function -------------------

def predict(text):
    if not model or not tokenizer:
        return None, None

    try:
        encoded = tokenizer(
            text,
            return_tensors='pt',
            padding='max_length',
            max_length=64,
            truncation=True
        )
        input_ids = encoded['input_ids']
        with torch.no_grad():
            logits = model(input_ids)
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            conf = probs[0][pred].item()
        return CLASS_LABELS[pred], conf
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None

# ------------------- UI -------------------

st.subheader("🔮 Enter News Text")
user_input = st.text_area("News Headline or Snippet", height=150)

if st.button("Predict") and user_input:
    with st.spinner("Classifying..."):
        label, confidence = predict(user_input)
    if label:
        st.success(f"Predicted: **{label}**")
        st.metric(label="Confidence", value=f"{confidence:.2%}")

st.caption("Built with Streamlit & PyTorch 💡")

