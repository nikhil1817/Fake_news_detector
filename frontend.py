import streamlit as st
import requests


st.title("Fake News Detector")
st.write("Enter a news article or headline to check whether it is fake or real.")

news_text = st.text_area("News Text", height=200)

if st.button("Predict"):
    if not news_text.strip():
        st.warning("Please enter some text.")
    else:
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json={"text": news_text}
        )

        if response.status_code == 200:
            result = response.json()

            if "error" in result:
                st.error(result["error"])
            else:
                st.subheader("Prediction")
                st.write(result["prediction"])

                st.subheader("Confidence")
                st.write(result["confidence"])
        else:
            st.error("Backend error. Make sure FastAPI is running.")
