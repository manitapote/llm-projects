import streamlit as st
import requests

st.set_page_config(
    page_title="Hate Speech Detector",
    page_icon="🔍",
    layout="centered"
)

st.title("🔍 Hate Speech Detection")
st.markdown("Enter text below to analyze whether it is targeted hate speech toward religious, ethnic, and political groups (muslims and jews).")

API_URL = "http://localhost:8000/predict"

text_input = st.text_area(
    "Enter text to analyze:",
    placeholder="Type or paste text here...",
    height=150
)

if st.button("Analyze", type="primary"):
    if not text_input.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Analyzing..."):
            try:
                response = requests.post(
                    API_URL,
                    json={"text": text_input},
                    timeout=30
                )
                result = response.json()
                label = result["label"]
                confidence = result["confidence"]

                st.markdown("---")
                st.subheader("Result")

                if label == "hate_speech":
                    st.error("⚠️ Hate Speech Detected")
                else:
                    st.success("✅ No Hate Speech Detected")

                st.metric("Confidence", f"{confidence:.1%}")
                st.progress(confidence)

                with st.expander("See details"):
                    st.json(result)

            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API. Make sure FastAPI is running on port 8000.")
            except Exception as e:
                st.error(f"Error: {str(e)}")

with st.sidebar:
    st.header("About")
    st.markdown("""
    Fine-tuned **BERT** model for hate speech detection.
    
    **Model Performance:**
    - F1 Score (Hate): 0.826
    - Macro F1: 0.874
    
    **Config:**
    - Learning Rate: 3e-05
    - Epochs: 3
    - Batch Size: 32
    """)
    st.markdown("Built by Manita Pote")
    st.markdown("[GitHub](https://github.com/manitapote/llm-projects/tree/main/bert_hatespeech)")