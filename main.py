import streamlit as st
import pickle
import nltk
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn
from nltk.stem.porter import PorterStemmer

st.set_page_config(page_title='Spam classifier')
tfidf = pickle.load(open(r'C:\Users\Rakesh\PycharmProjects\spam classifier\vectoriser.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))
ps= PorterStemmer()
st.title('Email/SMS Spam classifier')

import string


def transform_text(text):
    text = text.lower()

    # tokenization
    text = nltk.word_tokenize(text)

    y = []

    # remove special chars
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y.copy()
    y.clear()
    # remove punctuation and stopwords
    for i in text:
        if i not in string.punctuation and i not in nltk.corpus.stopwords.words('english'):
            y.append(i)

    text = y.copy()
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)
input_text = st.text_area('Enter the text: ')


if st.button('Predict'):
    # 1. preprocess
    transformed_input = transform_text(input_text)

    # 2. vectorize
    vectorized_input = tfidf.transform([transformed_input])

    # 3. predict
    prediction = model.predict(vectorized_input)[0]

    # 4. display
    if prediction == 1:
        st.header('spam')
    else:
        st.header('Not spam')