import streamlit as st
from decision_engine import text_decision, image_decision

st.set_page_config(page_title="Email Spam AI System", layout="centered")

st.title("📧 Email Spam AI Visual Security & Behavior Analysis")

option = st.radio("Choose Input Type:", ["Text Email", "Image Email"])

if option == "Text Email":
    text = st.text_area("Paste Email Content")

    if st.button("Analyze Text"):
        label, conf, source = text_decision(text)
        st.success(f"Result: {label.upper()}")
        st.write(f"Confidence: {conf:.2f}")
        st.write(f"Decision Source: {source}")

else:
    image = st.file_uploader("Upload Email Image", type=["png", "jpg", "jpeg"])

    if image:
        with open("temp_image.png", "wb") as f:
            f.write(image.read())

        label, conf, source = image_decision("temp_image.png")
        st.image("temp_image.png", caption="Uploaded Image")
        st.success(f"Result: {label.upper()}")
        st.write(f"Confidence: {conf:.2f}")
        st.write(f"Decision Source: {source}")
