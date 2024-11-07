import streamlit as st

st.title("Disease Prediction App")

text_input = st.text_input("Enter text")

if text_input is not None:
    st.button("Hit")
    st.write("Output")

    s