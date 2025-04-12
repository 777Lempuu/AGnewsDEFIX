import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Configure page
st.set_page_config(page_title="AG News Classifier", layout="wide")
st.title("📰 AG News Headline Classifier")

# Class labels mapping
CLASS_LABELS = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech"
}

@st.cache_resource
def load_model():
    try:
        device = torch.device('cpu')
        
        # Initialize model architecture
        model = AutoModelForSequenceClassification.from_pretrained(
            "google/bert_uncased_L-2_H-128_A-2",
            num_labels=4
        )
        
        # Load just the weights (what you actually have)
        weights = torch.load('AG_DeFix.pt', map_location=device)
        
        # Create a state_dict that matches the model's expectations
        state_dict = {
            'bert.embeddings.word_embeddings.weight': weights['embedding.weight'],
            'classifier.weight': weights['fc.weight'],
            'classifier.bias': weights['fc.bias']
        }
        
        # Load the modified state_dict
        model.load_state_dict(state_dict, strict=False)
        model.to(device).eval()
        
        # Initialize tokenizer separately
        tokenizer = AutoTokenizer.from_pretrained("google/bert_uncased_L-2_H-128_A-2")
        
        st.success("✅ Model loaded successfully!")
        return model, tokenizer
        
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        return None, None

# Load model
model, tokenizer = load_model()

def predict(text):
    """Make prediction with proper error handling"""
    if not model or not tokenizer:
        return None, None
        
    try:
        inputs = tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors="pt"
        )
        with torch.no_grad():
            outputs = model(**inputs).logits
        probs = torch.nn.functional.softmax(outputs, dim=1)
        pred_class = torch.argmax(probs).item()
        confidence = torch.max(probs).item()
        return CLASS_LABELS[pred_class], confidence
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None

# --- UI Components ---

# Dataset preview (optional)
@st.cache_data
def load_sample_data():
    url = "https://drive.google.com/uc?id=1xr-eyagU6GeZlYpn8qGIuMSdK5WFUV5x"
    return pd.read_csv(url).dropna()

if st.checkbox("Show sample dataset"):
    df = load_sample_data()
    num_rows = st.slider("Rows to display", 5, 100, 10)
    st.dataframe(df.head(num_rows))

# Main prediction interface
st.subheader("🔮 News Classifier")
user_input = st.text_area("Enter news text:", height=150)

if st.button("Predict") and user_input:
    with st.spinner("Analyzing..."):
        category, confidence = predict(user_input)
        
    if category:
        st.success(f"Predicted Category: **{category}**")
        st.metric("Confidence", f"{confidence:.1%}")
        
        # Optional: Show explanation
        with st.expander("What does this mean?"):
            st.markdown(f"""
            The model believes this text belongs to **{category}** news with {confidence:.1%} confidence.
            
            * 0: World 🌍
            * 1: Sports ⚽
            * 2: Business 💼  
            * 3: Sci/Tech 🔬
            """)

# Footer
st.markdown("---")
st.caption("Built with 🤗 Transformers and Streamlit")
