import streamlit as st
from PyPDF2 import PdfReader
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import nltk

chunks = [[]]
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

sep = "------------------------------------------------------------------------------------------------------------------"

for i in range(number_of_pages):
    page = reader.pages[i]
    text_to_sumarize += page.extract_text()

sentences = nltk.word_tokenize(text_to_sumarize)
print(sentences)

#we now need to separate sentence into chunks of about 480 tokens
i=0
for word in sentences:
    if len(chunks[i])<480:
        chunks[i].append(word)
    else:
        chunks += [[]]
        i += 1
        print(i)
        chunks[i].append(word)

st.write(i)

chunks_strings = [""]*(i+1)

for j in range(len(chunks)):
    chunks_strings[j] = " ".join(chunks[j])

model_name= "sshleifer/distilbart-cnn-12-6"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

for chunk in chunks_strings:

    batch = tokenizer(chunk, padding=True, truncation=True, max_length=1024 , return_tensors="pt")
    print(batch['input_ids'])
    print(sep)
    print(sep)
    with torch.no_grad():
        output = model.generate(batch['input_ids'], max_length=200,num_beams=4)
        print(output)
        print(sep)
        result = tokenizer.decode(output[0], skip_special_tokens=True)
        print(result)
        st.write(result)

st.markdown("_DONE_")
