import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import gdown
import os
import pickle

# Set up the app
st.set_page_config(page_title="AG News Classifier via SAFESTUDENT", page_icon="📰")
st.title("AG News Classifier via SAFESTUDENT")
st.write("Classify news articles into World, Sports, Business, or Sci/Tech categories")

# Configuration
MODEL_URL = "https://drive.google.com/uc?id=1GFir7sAkaxLXLeCsPpBE_UBb8wfJlnyX"
VOCAB_URL = "https://drive.google.com/uc?id=1XGpebvsOQOxuZLZR3Vf4giZ-rX_8dRSN"
MODEL_PATH = "AG_SafeStudent.pt"
VOCAB_PATH = "ag_news_vocab.pkl"

# Model architecture
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        emb = self.embedding(x)
        pooled = emb.mean(dim=1)
        return self.fc(pooled)

# File download function
def download_file(url, output):
    if not os.path.exists(output):
        try:
            with st.spinner(f"Downloading {output}..."):
                gdown.download(url, output, quiet=False)
            return True
        except Exception as e:
            st.error(f"Failed to download {output}: {e}")
            return False
    return True

# Load vocabulary
def load_vocabulary():
    if not download_file(VOCAB_URL, VOCAB_PATH):
        return None
    
    try:
        with open(VOCAB_PATH, 'rb') as f:
            vocab = pickle.load(f)
        return vocab
    except Exception as e:
        st.error(f"Vocabulary loading failed: {e}")
        return None

# Load model with correct vocabulary size
def load_model():
    if not download_file(MODEL_URL, MODEL_PATH):
        return None
    
    vocab = load_vocabulary()
    if vocab is None:
        return None
    
    try:
        # Initialize with correct vocabulary size
        model = TextClassifier(len(vocab), embed_dim=64, num_classes=4)
        
        # Load state dict
        model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None

# Load resources
vocab = load_vocabulary()
model = load_model()

# Text processing
def tokenize(text):
    return text.lower().split()

def text_pipeline(text):
    return [vocab.get(tok, vocab["<unk>"]) for tok in tokenize(text)]

def predict(text):
    if model is None:
        return "Model not loaded", []
    
    try:
        tokens = text_pipeline(text)
        tokens_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
        
        with torch.no_grad():
            logits = model(tokens_tensor)
            probs = torch.softmax(logits, dim=1).squeeze().numpy()
            pred_class = logits.argmax().item()
        
        class_names = ['World', 'Sports', 'Business', 'Sci/Tech']
        return class_names[pred_class], probs
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return "Error", []

# UI
user_input = st.text_area("Enter news text:", "Apple announced new products today.")

if st.button("Classify"):
    if user_input.strip():
        prediction, probs = predict(user_input)
        
        if prediction != "Error":
            st.success(f"Category: {prediction}")
            
            prob_df = pd.DataFrame({
                'Category': ['World', 'Sports', 'Business', 'Sci/Tech'],
                'Probability': probs
            })
            st.bar_chart(prob_df.set_index('Category'))

# Sidebar
st.sidebar.markdown("""
### About
Classifies news into:
- World
- Sports
- Business
- Sci/Tech
""")

if model is None:
    st.sidebar.error("Model not loaded")
else:
    st.sidebar.success("Model ready")
