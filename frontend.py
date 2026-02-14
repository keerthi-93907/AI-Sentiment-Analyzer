import streamlit as st
import requests
import json

# --- CONFIG ---
BACKEND_URL = "http://127.0.0.1:8000/predict"

# --- PAGE SETUP ---
st.set_page_config(page_title="AI Sentiment Analyzer", page_icon="🧠")

st.title("🧠 Neural Network Sentiment Analyzer")
st.markdown("""
This app uses a **Deep Learning (LSTM)** model to analyze the sentiment of your text.
Enter a movie review, tweet, or sentence below!
""")

# --- USER INPUT ---
user_text = st.text_area("Enter text here:", height=150, placeholder="e.g., The movie was absolutely fantastic! The acting was great.")

# --- HELPERS ---
def highlight_words(text, sentiment):
    # Simple heuristic visualization
    # In a real app, this would use attention weights from the model
    positive_words = ["good", "great", "fantastic", "amazing", "love", "loved", "excellent", "best", "beautiful"]
    negative_words = ["bad", "terrible", "worst", "hate", "hated", "awful", "boring", "slow", "poor"]
    
    words = text.split()
    highlighted_text = ""
    
    for word in words:
        clean_word = word.lower().strip(".,!?")
        if sentiment == "Positive" and clean_word in positive_words:
            highlighted_text += f"<span style='background-color: #90ee90; padding: 2px; border-radius: 4px; color: black'>{word}</span> "
        elif sentiment == "Negative" and clean_word in negative_words:
            highlighted_text += f"<span style='background-color: #ffcccb; padding: 2px; border-radius: 4px; color: black'>{word}</span> "
        else:
            highlighted_text += f"{word} "
            
    return highlighted_text

# --- PREDICTION LOGIC ---
if st.button("Analyze Sentiment"):
    if user_text:
        with st.spinner("Analyzing..."):
            try:
                # Send to Backend
                payload = {"text": user_text}
                response = requests.post(BACKEND_URL, json=payload)
                
                if response.status_code == 200:
                    result = response.json()
                    sentiment = result['sentiment']
                    confidence = result['confidence']
                    
                    # --- DISPLAY RESULTS ---
                    st.divider()
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if sentiment == "Positive":
                            st.success(f"## 😊 Positive")
                        else:
                            st.error(f"## 😠 Negative")
                    
                    with col2:
                        st.metric("Confidence Score", f"{confidence:.2%}")
                    
                    st.subheader("Text Highlights")
                    # Render HTML for highlighting
                    html_text = highlight_words(user_text, sentiment)
                    st.markdown(html_text, unsafe_allow_html=True)
                    
                else:
                    st.error(f"Error from API: {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("❌ Could not connect to the Backend API. Is it running?")
    else:
        st.warning("Please enter some text first.")

# --- SIDEBAR INFO ---
st.sidebar.title("About")
st.sidebar.info(
    """
    **Architecture:**
    1. Frontend: Streamlit
    2. Backend: FastAPI
    3. Model: LSTM (Keras/TensorFlow)
    
    **Dataset:** IMDB Reviews
    """
)   