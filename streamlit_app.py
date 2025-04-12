import streamlit as st
import torch
import torch.nn as nn
import re

# Recovered architecture
class RecoveredNewsClassifier(nn.Module):
    def __init__(self, vocab_size=65045, embed_dim=64, num_classes=4):
        super(RecoveredNewsClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, input_ids):
        embeds = self.embedding(input_ids)
        pooled = embeds.mean(dim=1)
        return self.fc(pooled)

# Simple tokenizer using word to index mapping
def basic_tokenizer(text, vocab, max_len=50):
    text = re.sub(r'[^\w\s]', '', text.lower())
    tokens = text.split()
    ids = [vocab.get(tok, vocab['[UNK]']) for tok in tokens[:max_len]]
    if len(ids) < max_len:
        ids += [vocab['[PAD]']] * (max_len - len(ids))
    return torch.tensor([ids])

# Dummy vocab (you can replace this with a saved one if available)
def create_dummy_vocab(vocab_size):
    vocab = {f"word{i}": i for i in range(4, vocab_size)}
    vocab['[PAD]'] = 0
    vocab['[UNK]'] = 1
    vocab['[CLS]'] = 2
    vocab['[SEP]'] = 3
    return vocab

CLASS_LABELS = {
    0: "World 🌍",
    1: "Sports ⚽",
    2: "Business 💼",
    3: "Sci/Tech 🔬"
}

st.title("📰 AG News Classifier (DeFix .pt)")

@st.cache_resource
def load_model():
    model = RecoveredNewsClassifier()
    model.load_state_dict(torch.load("AG_DeFix.pt", map_location='cpu'))
    model.eval()
    return model

model = load_model()
vocab = create_dummy_vocab(65045)

text = st.text_area("Enter a news headline:", height=150)

if st.button("Classify"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        input_ids = basic_tokenizer(text, vocab)
        with torch.no_grad():
            logits = model(input_ids)
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred].item()
        st.success(f"Prediction: **{CLASS_LABELS[pred]}**")
        st.metric("Confidence", f"{confidence:.2%}")


