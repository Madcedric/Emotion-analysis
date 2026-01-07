import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

# ===============================
# Load models & encoders
# ===============================
emotion_model = joblib.load("emotion_model.pkl")
emotion_vectorizer = joblib.load("emotion_vectorizer.pkl")
emotion_le = joblib.load("emotion_label_encoder.pkl")

depression_model = joblib.load("depression_model.pkl")
depression_vectorizer = joblib.load("depression_vectorizer.pkl")

# ===============================
# Streamlit UI
# ===============================
st.set_page_config(
    page_title="Emotion & Depression Prediction",
    layout="centered"
)

st.title("🧠 Emotion & Depression Prediction")
st.write("Enter a sentence to analyze **emotion** and **depression status**.")

text_input = st.text_area(
    "✍️ Enter text here:",
    placeholder="Type your sentence here..."
)

# ===============================
# Prediction
# ===============================
if st.button("🔍 Analyze") and text_input.strip():

    text = text_input.lower().strip()

    # -------- Emotion Prediction --------
    text_vec_emotion = emotion_vectorizer.transform([text])
    emotion_probs = emotion_model.predict_proba(text_vec_emotion)[0]
    emotion_index = np.argmax(emotion_probs)
    emotion_label = emotion_le.inverse_transform([emotion_index])[0]
    emotion_confidence = emotion_probs[emotion_index] * 100

    # -------- Depression Prediction --------
    text_vec_dep = depression_vectorizer.transform([text])
    dep_probs = depression_model.predict_proba(text_vec_dep)[0]
    dep_pred = np.argmax(dep_probs)
    dep_confidence = dep_probs[dep_pred] * 100

    # ===============================
    # Results
    # ===============================
    st.subheader("📊 Results")

    st.success(
        f"🎭 **Predicted Emotion:** {emotion_label.upper()} "
        f"({emotion_confidence:.2f}% confidence)"
    )

    if dep_pred == 1:
        st.error(
            f"⚠️ **Depression Detected** "
            f"({dep_confidence:.2f}% confidence)"
        )
    else:
        st.success(
            f"✅ **No Depression Detected** "
            f"({dep_confidence:.2f}% confidence)"
        )

    # ===============================
    # Emotion Probability Chart
    # ===============================
    st.subheader("📈 Emotion Probability Distribution")

    fig, ax = plt.subplots()
    ax.bar(
        emotion_le.classes_,
        emotion_probs
    )
    ax.set_ylabel("Probability")
    ax.set_xlabel("Emotion")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=40)

    st.pyplot(fig)

else:
    st.info("Enter a sentence and click **Analyze**")
