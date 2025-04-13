import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import gdown
import os
from collections import defaultdict
import pickle

# Set up the app
st.set_page_config(page_title="AG News Classifier", page_icon="📰")
st.title("AG News Classifier")
st.write("Classify news articles into World, Sports, Business, or Sci/Tech categories")

# Configuration - Updated with your Google Drive link
MODEL_URL = "https://drive.google.com/uc?id=1GFir7sAkaxLXLeCsPpBE_UBb8wfJlnyX"
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

# File download function with progress bar
@st.cache_resource
def download_file(url, output):
    if not os.path.exists(output):
        try:
            with st.spinner(f"Downloading {output}..."):
                gdown.download(url, output, quiet=False)
            st.success(f"Downloaded {output}")
            return True
        except Exception as e:
            st.error(f"Failed to download {output}: {e}")
            return False
    return True

# Initialize vocabulary (fallback version)
@st.cache_resource
def load_vocabulary():
    # Try to load vocabulary file if it exists
    if os.path.exists(VOCAB_PATH):
        try:
            with open(VOCAB_PATH, 'rb') as f:
                return pickle.load(f)
        except:
            st.warning("Failed to load vocabulary file, using fallback")
    
    # Fallback minimal vocabulary (should replace with your actual vocab)
    st.warning("Using fallback vocabulary - for best results, provide a vocabulary file")
    vocab = defaultdict(lambda: 1)  # Default to <unk>
    vocab.update({
        "<pad>": 0,
        "<unk>": 1,
        # Add some common words that might appear in AG News
        "the": 2, "of": 3, "to": 4, "and": 5, "in": 6, "a": 7, "for": 8,
        "on": 9, "is": 10, "that": 11, "by": 12, "this": 13, "with": 14,
        "as": 15, "at": 16, "from": 17, "be": 18, "are": 19, "has": 20
    })
    return vocab

# Load the model
@st.cache_resource
def load_model():
    if not download_file(MODEL_URL, MODEL_PATH):
        st.error("Model download failed - cannot continue")
        return None
    
    try:
        vocab = load_vocabulary()
        model = TextClassifier(len(vocab), embed_dim=64, num_classes=4)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        model.eval()
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
    return [vocab[tok] for tok in tokenize(text)]

def predict(text):
    if model is None:
        return "Model not loaded", []
    
    try:
        # Process input text
        tokens = text_pipeline(text)
        if not tokens:  # Handle empty input
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
        prediction, probabilities = predict(user_input)
        
        if prediction == "Error":
            st.error("Classification failed")
        else:
            st.subheader("Prediction Result")
            st.success(f"**Category:** {prediction}")
            
            # Display confidence scores
            st.subheader("Confidence Scores")
            classes = ['World', 'Sports', 'Business', 'Sci/Tech']
            prob_df = pd.DataFrame({
                'Category': classes,
                'Probability': probabilities
            })
            
            # Show both bar chart and table
            st.bar_chart(prob_df.set_index('Category'))
            st.table(prob_df.style.format({'Probability': '{:.2%}'}))
    else:
        st.warning("Please enter some text to classify")

# App information
st.sidebar.markdown("""
### About this app
This app uses a SafeStudent-trained model to classify news articles into 4 categories:
- **World** - International news and events
- **Sports** - Sports news and competitions
- **Business** - Business and financial news
- **Sci/Tech** - Science and technology news

The model was trained on the AG News dataset with knowledge distillation.
""")

if model is None:
    st.error("⚠️ Model failed to load. Please check:")
    st.error("- Internet connection")
    st.error("- Google Drive link accessibility")
    st.error("- File permissions")
