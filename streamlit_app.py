import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import gdown
import os
import pickle
from collections import defaultdict

# Set up the app
st.set_page_config(page_title="AG News Classifier", page_icon="📰")
st.title("AG News Classifier")
st.write("Classify news articles into World, Sports, Business, or Sci/Tech categories")

# Configuration - Updated with your actual files
MODEL_URL = "https://drive.google.com/uc?id=1GFir7sAkaxLXLeCsPpBE_UBb8wfJlnyX"
VOCAB_URL = "https://drive.google.com/uc?id=1XGpebvsOQOxuZLZR3Vf4giZ-rX_8dRSN"
MODEL_PATH = "AG_SafeStudent.pt"
VOCAB_PATH = "ag_news_vocab.pkl"

# Model architecture (must match your training code)
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        emb = self.embedding(x)
        pooled = emb.mean(dim=1)
        return self.fc(pooled)

# File download function with progress
@st.cache_resource
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

# Load vocabulary - now using your actual vocab file
@st.cache_resource
def load_vocabulary():
    # Download vocabulary file if needed
    if not download_file(VOCAB_URL, VOCAB_PATH):
        st.error("Failed to download vocabulary file")
        return None
    
    try:
        with open(VOCAB_PATH, 'rb') as f:
            vocab = pickle.load(f)
        st.success(f"Loaded vocabulary with {len(vocab)} words")
        return vocab
    except Exception as e:
        st.error(f"Vocabulary loading failed: {e}")
        return None

# Load the model with proper vocabulary
@st.cache_resource
def load_model():
    # First download the model
    if not download_file(MODEL_URL, MODEL_PATH):
        return None
    
    # Load vocabulary first to get correct size
    vocab = load_vocabulary()
    if vocab is None:
        return None
    
    try:
        model = TextClassifier(len(vocab), embed_dim=64, num_classes=4)
        model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
        model.eval()
        st.success("Model loaded successfully")
        return model
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

# Load resources
vocab = load_vocabulary()
model = load_model()

# Text processing functions
def tokenize(text):
    return text.lower().split()

def text_pipeline(text):
    return [vocab.get(tok, vocab["<unk>"]) for tok in tokenize(text)]

def predict(text):
    if model is None:
        return "Model not loaded", []
    
    try:
        # Process input text
        tokens = text_pipeline(text)
        if not tokens:
            return "Invalid input", []
            
        tokens_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
        
        # Predict
        with torch.no_grad():
            logits = model(tokens_tensor)
            probs = torch.softmax(logits, dim=1).squeeze().numpy()
            pred_class = logits.argmax().item()
        
        class_names = ['World', 'Sports', 'Business', 'Sci/Tech']
        return class_names[pred_class], probs
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return "Error", []

# User interface
user_input = st.text_area("Enter news text to classify:", 
                         "Apple announced new products at their annual developer conference.")

if st.button("Classify"):
    if user_input.strip():
        with st.spinner("Analyzing..."):
            prediction, probabilities = predict(user_input)
            
            if prediction == "Error":
                st.error("Classification failed")
            else:
                st.subheader("Prediction Result")
                st.success(f"**Category:** {prediction}")
                
                st.subheader("Confidence Scores")
                classes = ['World', 'Sports', 'Business', 'Sci/Tech']
                prob_df = pd.DataFrame({
                    'Category': classes,
                    'Probability': probabilities
                })
                
                # Display both visual and numeric results
                col1, col2 = st.columns(2)
                with col1:
                    st.bar_chart(prob_df.set_index('Category'))
                with col2:
                    st.table(prob_df.style.format({'Probability': '{:.2%}'}).highlight_max(axis=0))
    else:
        st.warning("Please enter some text to classify")

# App information
st.sidebar.markdown("""
### About this app
This app classifies news articles into 4 categories using a SafeStudent-trained model.

**Model Info:**
- Trained on AG News dataset
- Vocabulary size: 158,735 words
- Embedding dimension: 64

**Categories:**
- World 🌍
- Sports ⚽
- Business 💼
- Sci/Tech 🔬
""")

# System status
st.sidebar.markdown("""
### System Status
""")
if model is None:
    st.sidebar.error("❌ Model not loaded")
else:
    st.sidebar.success("✅ Model loaded")
    
if vocab is None:
    st.sidebar.error("❌ Vocabulary not loaded")
else:
    st.sidebar.success(f"✅ Vocabulary loaded ({len(vocab)} words)")

# Requirements instructions
st.sidebar.markdown("""
### Requirements
```text
streamlit
torch
numpy
pandas
gdown
