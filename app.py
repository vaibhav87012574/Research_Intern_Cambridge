import streamlit as st
from transformers import pipeline
import torch
from pathlib import Path  # <-- 1. Import pathlib

# --- 1. Load Your Fine-Tuned Model ---

# --- 2. Build a robust path ---
# This finds the directory containing app.py and joins it with 'motivation_model'
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model"

@st.cache_resource
def load_model():
    """
    Loads the fine-tuned model from the local directory.
    """
    print(f"Loading model from: {MODEL_PATH}") # This will print the full path
    
    # This must match the 'output_dir' from your training script
    model_path = MODEL_PATH # <-- 3. Use the new path
    
    # Use GPU if available (device=0), otherwise CPU (device=-1)
    device = 0 if torch.cuda.is_available() else -1
    
    classifier = pipeline(
        "text-classification", 
        model=model_path,
        device=device 
    )
    print("Model loaded successfully!")
    return classifier

# Load the model and handle potential errors
try:
    classifier = load_model()
except Exception as e:
    st.error(f"Error: Could not load model.")
    st.error(f"Please make sure the './motivation_model' directory exists in the same folder as app.py.")
    st.stop()


# --- 2. Set Up The Streamlit Interface ---

st.title("Motivation Analyzer 🧠")
st.markdown(
    "This app uses the `distilbert-base-uncased` model you fine-tuned to "
    "classify text as **intrinsic** or **extrinsic**."
)

# --- 3. Create the Input Text Area ---

# Create a text area for user input
user_text = st.text_area(
    "Enter a sentence to analyze:", 
    "I just want to get a good grade in this class."
)

# --- 4. Create the 'Analyze' Button and Logic ---

if st.button("Analyze"):
    if user_text:
        # If text is provided, run the model
        with st.spinner("Analyzing..."):
            result = classifier(user_text)
        
        # Extract the prediction and score
        prediction = result[0]['label']
        score = result[0]['score']

        # --- 5. Display the Result ---
        st.subheader("Analysis Result")
        
        # Show a different style for each prediction
        if prediction == 'intrinsic':
            st.success(f"Prediction: Intrinsic")
            st.write(
                "This text suggests internal motivation (e.g., passion, curiosity, personal growth)."
            )
        else: # Extrinsic
            st.info(f"Prediction: Extrinsic")
            st.write(
                "This text suggests external motivation (e.g., money, grades, praise, avoiding punishment)."
            )
        
        # Display the confidence score
        st.metric(label="Confidence", value=f"{score:.2%}")

        # Show the raw output in an expander
        with st.expander("Show raw model output"):
            st.json(result)
            
    else:
        # If no text is provided
        st.warning("Please enter some text to analyze.")


# Add a sidebar for more info
st.sidebar.header("About This App")
st.sidebar.write(
    "This app is the final step of the fine-tuning process we discussed. "
    "It loads the saved model from `./motivation_model` and uses it "
    "for real-time inference."
)