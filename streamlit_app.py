import streamlit as st
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Unduh resource NLTK
nltk.download('stopwords')

# === Fungsi Preprocessing ===
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

def preprocessing(teks):
    teks = teks.lower()
    teks = teks.translate(str.maketrans('', '', string.punctuation))
    tokens = teks.split()
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

# === Judul App ===
st.set_page_config(page_title="Sentiment Analysis App", layout="centered")
st.title("💬 Sentiment Analysis with Logistic Regression & TF-IDF")
st.write("Upload file CSV (dengan kolom `x` dan `y`) untuk analisis sentimen.")

# === Upload Dataset ===
uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['x'] = df['x'].astype(str)
    df['clean'] = df['x'].apply(preprocessing)

    st.subheader("📊 Statistik Dataset")
    st.write(df[['x', 'y']].head())

    label_count = df['y'].value_counts()
    st.write("Distribusi Label:")
    st.bar_chart(label_count)

    # === Wordcloud ===
    st.subheader("☁️ Wordcloud")
    tab1, tab2 = st.tabs(["Positif", "Negatif"])
    with tab1:
        words = " ".join(df[df['y'] == 1]['clean'])
        wc = WordCloud(width=800, height=400, background_color="white").generate(words)
        plt.imshow(wc, interpolation='bilinear')
        plt.axis("off")
        st.pyplot(plt.gcf())

    with tab2:
        words = " ".join(df[df['y'] == 0]['clean'])
        wc = WordCloud(width=800, height=400, background_color="black", colormap='Pastel1').generate(words)
        plt.imshow(wc, interpolation='bilinear')
        plt.axis("off")
        st.pyplot(plt.gcf())

    # === Model Training ===
    X_train, X_test, y_train, y_test = train_test_split(df['clean'], df['y'], test_size=0.2, random_state=42)

    tfidf = TfidfVectorizer()
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)

    model = LogisticRegression()
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)

    # === Confusion Matrix ===
    st.subheader("📉 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negatif", "Positif"], yticklabels=["Negatif", "Positif"])
    plt.xlabel("Prediksi")
    plt.ylabel("Aktual")
    st.pyplot(fig)

    # === Classification Report ===
    st.subheader("📑 Classification Report")
    report = classification_report(y_test, y_pred, target_names=["Negatif", "Positif"], output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    # === Prediksi Kalimat Baru ===
    st.subheader("📝 Prediksi Kalimat Baru")
    input_kalimat = st.text_area("Masukkan kalimat review:")
    if st.button("Prediksi"):
        kalimat_clean = preprocessing(input_kalimat)
        kalimat_vec = tfidf.transform([kalimat_clean])
        hasil = model.predict(kalimat_vec)[0]
        sentimen = "Positif ✅" if hasil == 1 else "Negatif ❌"
        st.success(f"Prediksi Sentimen: **{sentimen}**")
