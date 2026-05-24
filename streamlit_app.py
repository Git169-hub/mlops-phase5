import streamlit as st
import requests

st.title("RAG Question Answering")
st.write("Ask anything about MLOps, Docker, FastAPI, or RAG.")

question = st.text_input("Your Question")

if st.button("Ask"):
    if question.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            response = requests.post(
                "http://api:8000/ask",
                json={"question": question}
            )
            if response.status_code == 200:
                answer = response.json()["answer"]
                st.success(answer)
            else:
                st.error("API error. Is your Docker container running?")
