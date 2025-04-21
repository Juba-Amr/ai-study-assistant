import streamlit as st
from PyPDF2 import PdfReader
from transformers import pipeline



number_of_pages = 0
text_to_sumarize = ""
summary = ""
st.title("Welcome to your AI assisstant")

with st.form(key="get_pdf"):
    st.header("Upload your file here")
    uploaded_file = st.file_uploader(label='upload pdf', type='pdf')

    st.form_submit_button("done")

if uploaded_file != None :
    with st.form(key="select_pages"):
        reader = PdfReader(uploaded_file)
        max_pages = reader._get_num_pages()
        number_of_pages = st.slider(label='select how many pages you want to summarize',min_value=0, max_value=max_pages, step=1)

        st.form_submit_button("done")

for i in range(number_of_pages):
    page = reader.pages[i]
    text_to_sumarize += page.extract_text()



summarizer = pipeline(
    "summarization",
    model="google/pegasus-xsum",
    do_sample=False
)

chunks = len(text_to_sumarize)//500
for i in range(0,chunks):
    summary += summarizer(
        text_to_sumarize[i*500:(i+1)*500],
        min_length=20,
        max_length=30
    )[0]['summary_text']

st.write(summary)
