import os
import random
import openai
import PyPDF2
import pandas as pd
import sqlite3
import json
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')

def extract_text_from_pdf(file_path: str) -> str:
    print('hi 24')
    text = ""
    print('25')
    with open(file_path, "rb") as f:
        pdf_reader = PyPDF2.PdfReader(f)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    print(text)
    return text

def extract_resume_info(resume_text, api_key):
    prompt = f"""
    Это текст резюме. Ответ верни в виде "ответ;ответ;ответ" - без каких-либо еще знаков. Найди в нем информацию, описанную ниже и верни в нужном виде :
    город, в котором была последняя работа - только город
    название последней должности - только название
    общий опыт работы, в годах - только числа

    {resume_text}
    """
    openai.api_key = api_key
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    result = response['choices'][0]['message']['content'].strip()
    return result


with open('../final/data/keys.txt', 'r') as file:
    key_skills_list = file.read().lower().split("\n")

def extract_key_skills(resume_text):
    stop_words = set(nltk.corpus.stopwords.words('russian')).union(nltk.corpus.stopwords.words('english'))
    word_tokens = nltk.tokenize.word_tokenize(resume_text)
    filtered_tokens = [w for w in word_tokens if (w not in stop_words and  w.isalpha())]
    bigrams_trigrams = list(map(' '.join, nltk.everygrams(filtered_tokens, 2, 3)))
    key_skills_result = set()
    for token in filtered_tokens:
        if token.lower() in key_skills_list:
            key_skills_result.add(token)
    for ngram in bigrams_trigrams:
        if ngram.lower() in key_skills_list:
            key_skills_result.add(ngram)
    return list(key_skills_result)
