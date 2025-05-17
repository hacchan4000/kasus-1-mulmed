import streamlit as st
import pandas as pd
import re
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Load data
file = "Twitter_Emotion_Dataset.csv"
dataku = pd.read_csv(file)

# Encode labels
label_encoder = LabelEncoder()
label_encoded = label_encoder.fit_transform(dataku['label'])

# Stopwords
remover = StopWordRemoverFactory()
kata_stop = remover.get_stop_words()

# Cleaning function
def bersihkan_data(teks_series):
    return teks_series.apply(lambda teks: 
        re.sub(r'\[.*?\]', '', 
        re.sub(r'http\S+', '', 
        re.sub(r'\d+', '', 
        re.sub(r'[^a-zA-Z\s]', '', 
        teks.lower().strip()
    )))))

# Preprocessing pipeline
Pipeline_text = Pipeline([
    ('cleaning', FunctionTransformer(bersihkan_data, validate=False)),
    ('vectorizing', TfidfVectorizer(stop_words=kata_stop)),
])

# Fit and transform the tweets
dataku_transformed = Pipeline_text.fit_transform(dataku['tweet'])

# Split data
x_train_full, x_test, y_train_full, y_test = train_test_split(dataku_transformed, label_encoded, test_size=0.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train_full, y_train_full, test_size=0.2, random_state=42)



import math
import numpy as np
from collections import defaultdict
from sklearn.metrics import accuracy_score

# Custom Naive Bayes Classifier
class CustomNaiveBayes:
    def __init__(self): #constructor 
        
        self.class_priors = {} #nyimpen prior probability tiap class
        self.word_probs = {} #nyimpen probab tiap terhadap kelas p(kata|class)
        self.vocab_size = 0 #nomor kata unik
        self.class_word_totals = {} #total banyak kata tersebut muncul di tiap kelas
        self.class_word_counts = {} #frekuensi vektor per kelas

    def fit(self, X, y): #method untuk mengolah data train
        X = X.toarray() #ngubah sparse matriks dense matriks
        n_docs, n_features = X.shape #banyak tweet dan banyak kata unik
        self.vocab_size = n_features

        class_docs = defaultdict(list)
        for i in range(len(y)):#menggrubkan tweet berdasarkan label
            label = y.iloc[i] if hasattr(y, "iloc") else y[i]
            class_docs[label].append(X[i])

        self.class_priors = {c: len(docs) / n_docs for c, docs in class_docs.items()}#itung prior probabilitas

        for c, docs in class_docs.items():
            docs_array = np.array(docs)
            word_totals = docs_array.sum(axis=0)  # total frekuensi tiap kata dalam dokumen
            self.class_word_totals[c] = word_totals.sum()
            self.class_word_counts[c] = word_totals
            self.word_probs[c] = (word_totals + 1) / (self.class_word_totals[c] + self.vocab_size)  # Laplace smoothing

    def predict(self, X): #method untuk prediksi data
        X = X.toarray()
        preds = []

        for doc in X:
            class_scores = {}
            for c in self.class_priors:
                log_prob = math.log(self.class_priors[c])
                log_prob += np.sum(doc * np.log(self.word_probs[c]))
                class_scores[c] = log_prob
            preds.append(max(class_scores, key=class_scores.get))
        return preds

# Train model
model = CustomNaiveBayes()
model.fit(x_train, y_train)

# Define preprocessing for single tweet
def preprocess(teks):
    cleaned = re.sub(r'\[.*?\]', '', 
              re.sub(r'http\S+', '', 
              re.sub(r'\d+', '', 
              re.sub(r'[^a-zA-Z\s]', '', teks.lower().strip()))))
    return cleaned

# Vectorizer from pipeline (reuse)
vectorizer = Pipeline_text.named_steps['vectorizing']

# Streamlit UI
st.title("Emotion Detection from Tweet")
user_input = st.text_area("Masukkan tweet di sini:")

if st.button("Deteksi Emosi"):
    processed = preprocess(user_input)
    vektor = vectorizer.transform([processed])
    prediksi = model.predict(vektor)[0]
    label = label_encoder.inverse_transform([prediksi])[0]
    st.success(f"Emosi yang terdeteksi: {label.upper()}")
