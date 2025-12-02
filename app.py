

# import streamlit as st
# from streamlit_lottie import st_lottie
# import json
# import joblib

# # ---------------- Load Model + Encoder ----------------
# model = joblib.load("models/tfidf_logreg_best.joblib")
# label_encoder = joblib.load("models/label_encoder.joblib")

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
#         # Prediction
#         pred = model.predict([user_input])[0]
#         probs = model.predict_proba([user_input])[0]
#         label = label_encoder.inverse_transform([pred])[0]

#         # Output
#         if "bully" in label.lower():
#             result = f"<div class='output-box bully'>⚠️ Cyberbullying detected: {label}</div>"
#         else:
#             result = f"<div class='output-box safe'>✅ Safe message: {label}</div>"

#         st.markdown(result, unsafe_allow_html=True)
#         st.write("**Probabilities:**", dict(zip(label_encoder.classes_, probs)))
















############3------------------ DistilBert ----------------
# import streamlit as st
# from streamlit_lottie import st_lottie
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
# import torch
# import json

# # ---------------- Load Model ----------------
# @st.cache_resource
# def load_model():
#     model_name = "./models"  # change to your fine-tuned model if needed
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
#         # Find label with highest score
#         best = max(results, key=lambda x: x['score'])
#         label = best['label']
#         score = best['score']

#         if "NEGATIVE" in label:
#             result = f"<div class='output-box bully'>⚠️ Cyberbullying detected (Confidence: {score:.2f})</div>"
#         else:
#             result = f"<div class='output-box safe'>✅ Safe message (Confidence: {score:.2f})</div>"

#         st.markdown(result, unsafe_allow_html=True)

#         probs = {r['label']: round(r['score'], 3) for r in results}
#         st.write("**Probabilities:**", probs)















import streamlit as st
from streamlit_lottie import st_lottie
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import json

# ---------------- Load Model ----------------
@st.cache_resource
def load_model():
    model_name = "./models"  # path to your fine-tuned model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)
    return classifier

classifier = load_model()

# ---------------- CSS Styles ----------------
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(-45deg, #ff4b4b, #ff8c00, #4b7fff, #7d00ff);
        animation: gradientBG 15s ease infinite;
        height: 100vh;
    }

    @keyframes gradientBG {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }

    .title {
        font-size: 55px;
        font-weight: bold;
        background: linear-gradient(270deg, 
            #ff0000, #ff7f00, #ffff00, #00ff00, 
            #0000ff, #4b0082, #8f00ff, #ff0000);
        background-size: 1500% 1500%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: rainbow 10s linear infinite;
        margin-bottom: 10px;
    }

    @keyframes rainbow {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }

    .intro-line {
        font-size: 18px;
        color: white;
        margin-bottom: 25px;
        font-style:italic;
    }

    .stButton button {
        display: block;
        margin: 30px auto;
        background: linear-gradient(90deg, #ff4b4b, #ff8c00);
        color: white;
        border-radius: 12px;
        height: 3em;
        width: 12em;
        font-size: 18px;
        font-weight: bold;
        border: none;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.3);
        transition: 0.3s;
    }
    .stButton button:hover {
        transform: scale(1.1);
        background: linear-gradient(90deg, #7d00ff, #00d4ff);
    }

    .output-box {
        padding: 15px;
        border-radius: 12px;
        margin-top: 20px;
        font-size: 20px;
        font-weight: bold;
        text-align: center;
    }
    .safe {
        background-color: rgba(0,255,0,0.2);
        color: #00ff7f;
    }
    .bully {
        background-color: rgba(255,0,0,0.2);
        color: #ff4b4b;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- Load Lottie Animation ----------------
with open("Cybersecurity.json", "r") as f:
    lottie_anim = json.load(f)

# ---------------- Layout ----------------
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("<div class='title'>Cyberbullying Detection System</div>", unsafe_allow_html=True)
    st.markdown("<div class='intro-line'>This tool analyzes text messages and helps identify possible cyberbullying patterns.</div>", unsafe_allow_html=True)

with col2:
    st_lottie(lottie_anim, height=250, key="cyberbully")

# ---------------- Input Section ----------------
st.markdown("### Enter a message :")
user_input = st.text_area("", "")

# ---------------- Prediction Button ----------------
if st.button("🔍 Detect"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        results = classifier(user_input)[0]
        best = max(results, key=lambda x: x['score'])
        label = best['label']
        score = best['score']

        # Cyberbullying logic
        if label.lower() == "not_cyberbullying":
            result = f"<div class='output-box safe'>✅ Safe message (Confidence: {score:.2f})</div>"
        else:
            result = f"<div class='output-box bully'>⚠️ Cyberbullying detected ({label})<br>Confidence: {score:.2f}</div>"

        st.markdown(result, unsafe_allow_html=True)

        # Probability dictionary
        probs = {r['label']: round(r['score'], 3) for r in results}
        st.json(probs)
