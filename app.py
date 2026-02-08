# import streamlit as st
# from streamlit_lottie import st_lottie
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
# import torch
# import json

# # ---------------- Load Model ----------------
# @st.cache_resource
# def load_model():
#     model_name = "./models"  # path to your fine-tuned model
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForSequenceClassification.from_pretrained(model_name)
#     classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)
#     return classifier

# classifier = load_model()

# # ---------------- CSS Styles ----------------
# st.markdown(
#     """
#     <style>
#     body {
#         background: linear-gradient(-45deg, #ff4b4b, #ff8c00, #4b7fff, #7d00ff);
#         animation: gradientBG 15s ease infinite;
#         height: 100vh;
#     }

#     @keyframes gradientBG {
#         0% {background-position: 0% 50%;}
#         50% {background-position: 100% 50%;}
#         100% {background-position: 0% 50%;}
#     }

#     .title {
#         font-size: 55px;
#         font-weight: bold;
#         background: linear-gradient(270deg, 
#             #ff0000, #ff7f00, #ffff00, #00ff00, 
#             #0000ff, #4b0082, #8f00ff, #ff0000);
#         background-size: 1500% 1500%;
#         -webkit-background-clip: text;
#         -webkit-text-fill-color: transparent;
#         animation: rainbow 10s linear infinite;
#         margin-bottom: 10px;
#     }

#     @keyframes rainbow {
#         0% {background-position: 0% 50%;}
#         50% {background-position: 100% 50%;}
#         100% {background-position: 0% 50%;}
#     }

#     .intro-line {
#         font-size: 18px;
#         color: white;
#         margin-bottom: 25px;
#         font-style:italic;
#     }

#     .stButton button {
#         display: block;
#         margin: 30px auto;
#         background: linear-gradient(90deg, #ff4b4b, #ff8c00);
#         color: white;
#         border-radius: 12px;
#         height: 3em;
#         width: 12em;
#         font-size: 18px;
#         font-weight: bold;
#         border: none;
#         box-shadow: 0px 4px 10px rgba(0,0,0,0.3);
#         transition: 0.3s;
#     }
#     .stButton button:hover {
#         transform: scale(1.1);
#         background: linear-gradient(90deg, #7d00ff, #00d4ff);
#     }

#     .output-box {
#         padding: 15px;
#         border-radius: 12px;
#         margin-top: 20px;
#         font-size: 20px;
#         font-weight: bold;
#         text-align: center;
#     }
#     .safe {
#         background-color: rgba(0,255,0,0.2);
#         color: #00ff7f;
#     }
#     .bully {
#         background-color: rgba(255,0,0,0.2);
#         color: #ff4b4b;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# # ---------------- Load Lottie Animation ----------------
# with open("Cybersecurity.json", "r") as f:
#     lottie_anim = json.load(f)

# # ---------------- Layout ----------------
# col1, col2 = st.columns([2, 1])

# with col1:
#     st.markdown("<div class='title'>Cyberbullying Detection System</div>", unsafe_allow_html=True)
#     st.markdown("<div class='intro-line'>This tool analyzes text messages and helps identify possible cyberbullying patterns.</div>", unsafe_allow_html=True)

# with col2:
#     st_lottie(lottie_anim, height=250, key="cyberbully")

# # ---------------- Input Section ----------------
# st.markdown("### Enter a message :")
# user_input = st.text_area("", "")

# # ---------------- Prediction Button ----------------
# if st.button("🔍 Detect"):
#     if not user_input.strip():
#         st.warning("Please enter some text.")
#     else:
#         results = classifier(user_input)[0]
#         best = max(results, key=lambda x: x['score'])
#         label = best['label']
#         score = best['score']

#         # Cyberbullying logic
#         if label.lower() == "not_cyberbullying":
#             result = f"<div class='output-box safe'>✅ Safe message (Confidence: {score:.2f})</div>"
#         else:
#             result = f"<div class='output-box bully'>⚠️ Cyberbullying detected ({label})<br>Confidence: {score:.2f}</div>"

#         st.markdown(result, unsafe_allow_html=True)

#         # Probability dictionary
#         probs = {r['label']: round(r['score'], 3) for r in results}
#         st.json(probs)





import streamlit as st
import joblib

st.set_page_config(page_title="Cyberbullying Detector", layout="wide")

# Custom CSS
st.markdown("""
<style>
    body {
        background: linear-gradient(-45deg, #7d00ff, #5500ff, #4b7fff, #0099ff);
        background-size: 500% 500%;
    }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ Cyberbullying Detection System")
st.markdown("End-to-end NLP system for detecting cyberbullying, including religion-based abuse")

# Load model (cached - loads only once)
@st.cache_resource
def load_model():
    try:
        model = joblib.load("tfidf_logreg_best.jobilib")
        vectorizer = joblib.load("vocab")
        label_encoder = joblib.load("label_encoder.jobilib")
        return model, vectorizer, label_encoder
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

model, vectorizer, label_encoder = load_model()

if model is None:
    st.error("❌ Model files not found or corrupted")
else:
    # Input
    text = st.text_area("Enter text to analyze:", height=150, placeholder="Type a message here...")
    
    if st.button("🔍 Detect", use_container_width=True):
        if not text.strip():
            st.warning("⚠️ Please enter some text")
        else:
            try:
                # Vectorize
                text_vector = vectorizer.transform([text])
                
                # Predict
                prediction = model.predict(text_vector)[0]
                probabilities = model.predict_proba(text_vector)[0]
                score = max(probabilities)
                
                # Decode label
                label = label_encoder.inverse_transform([prediction])[0]
                
                # Category definitions
                cyberbullying_types = {
                    "age": {"emoji": "👶", "text": "Age-Based Cyberbullying"},
                    "gender": {"emoji": "⚥️", "text": "Gender-Based Cyberbullying"},
                    "ethnicity": {"emoji": "🌍", "text": "Ethnicity-Based Cyberbullying"},
                    "religion": {"emoji": "🙏", "text": "Religion-Based Cyberbullying"},
                    "other_cyberbullying": {"emoji": "⚠️", "text": "Other Cyberbullying Detected"},
                    "not_cyberbullying": {"emoji": "✅", "text": "Safe Message"}
                }
                
                # Get category info
                label_lower = str(label).lower().strip()
                category = cyberbullying_types.get(label_lower, cyberbullying_types.get("not_cyberbullying"))
                
                # Display result
                st.markdown("---")
                
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    st.markdown(f"<h1>{category['emoji']}</h1>", unsafe_allow_html=True)
                
                with col2:
                    if label_lower == "not_cyberbullying":
                        st.success(f"**{category['text']}**")
                        st.write(f"✅ Confidence: {score:.2%}")
                    else:
                        st.error(f"**{category['text']}**")
                        st.write(f"⚠️ Category: **{label}**")
                        st.write(f"Confidence: {score:.2%}")
                
                # Progress bar
                st.progress(score)
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
