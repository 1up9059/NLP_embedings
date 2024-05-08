import pdb
import tensorflow as tf
import re
import string
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import PyPDF2
from pathlib import Path
import multiprocessing as ml
from gensim.models import Word2Vec

def read_pdf(file_path):
    # Read and extract text from a PDF file
    with open(file_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() if page.extract_text() else ""
    return text
    
def read_csv(file_path):
    # Read and extract text from a CSV file
    df = pd.read_csv(file_path)
    text = ' '. join(df['content'].astype(str))
    return text

def read_txt(file_path):
    with open(file_path, 'r') as file:
        text= file.read()
        return text

def get_path(folder_path):
    mpath = Path(folder_path)
    files_to_process = []
    for subfolder in mpath.iterdir():
        if subfolder.is_dir():
            files_to_process.extend(subfolder.glob('*.csv'))
            files_to_process.extend(subfolder.glob('*.pdf'))
            files_to_process.extend(subfolder.glob('*.txt'))
    return files_to_process

def get_text(texts):
    total_text = ''
    for path in texts:
        if path.suffix == '.csv':
            total_text += read_csv(path)
        elif path.suffix == '.pdf':
            total_text += read_pdf(path)
        elif path.suffix == '.txt':
            total_text += read_txt(path)    
    return total_text

def clean_sentences(sentences):
    clean_sentences = []
    for st in sentences:
        tokens = st.translate(str.maketrans('', '', string.punctuation)).split()
        tokens = [word.lower() for word in tokens if word.isalpha()]
        if tokens:
            clean_sentences.append(tokens)
    return clean_sentences

def model_word2vector(sentences):
    model500 = Word2Vec(sentences, vector_size=500, window=5, min_count= 4, workers=ml.cpu_count())
    model3 = Word2Vec(sentences, vector_size=3, window=5, min_count= 4, workers=ml.cpu_count())
    return model500, model3


def main():
    main_folder_path = 'data' 
    all_texts = get_path(main_folder_path)
    text = get_text(all_texts)
    sentences = text.split('.')
    pdb.set_trace()
    sentences = clean_sentences(sentences)
    model1500, model3 = model_word2vector(sentences)
    pdb.set_trace()
    
    
if __name__ == '__main__':
    main()
    

            