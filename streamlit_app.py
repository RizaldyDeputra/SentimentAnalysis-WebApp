import streamlit as st
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

# Unduh stopwords
nltk.download('stopwords')

# === PREPROCESSING ===
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocessing(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# === KONFIGURASI STREAMLIT ===
st.set_page_config(page_title="Analisis Sentimen", layout="centered")
st.title("💬 Analisis Sentimen dengan Logistic Regression + TF-IDF")
st.write("Unggah file CSV dengan kolom `review` dan `label`.")

# === UPLOAD DATASET ===
uploaded_file = st.file_uploader("Unggah file CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file, delimiter=';')
    
    # Pastikan kolom sesuai
    if 'review' not in df.columns or 'label' not in df.columns:
        st.error("CSV harus mengandung kolom 'review' dan 'label'")
    else:
        df['review'] = df['review'].astype(str)
        df['clean'] = df['review'].apply(preprocessing)

        # === Tampilkan data awal ===
        st.subheader("📄 Data Awal")
        st.write(df[['review', 'clean', 'label']].head())

        if df['label'].isnull().any():
            st.warning("Terdapat nilai kosong di kolom label. Baris tersebut akan dihapus.")
            df = df.dropna(subset=['label'])

        # === Distribusi label ===
        st.subheader("📊 Distribusi Label")
        col1, col2 = st.columns(2)
        with col1:
            st.write(df['label'].value_counts())
        with col2:
            st.bar_chart(df['label'].value_counts())

        # === Wordcloud ===
        st.subheader("☁️ Wordcloud")
        tab1, tab2 = st.tabs(["Positif", "Negatif"])
        with tab1:
            words = ' '.join(df[df['label'] == 1]['clean'])
            wc = WordCloud(width=800, height=400, background_color="white").generate(words)
            plt.imshow(wc, interpolation='bilinear')
            plt.axis("off")
            st.pyplot(plt.gcf())
        with tab2:
            words = ' '.join(df[df['label'] == 0]['clean'])
            wc = WordCloud(width=800, height=400, background_color="black", colormap='Pastel1').generate(words)
            plt.imshow(wc, interpolation='bilinear')
            plt.axis("off")
            st.pyplot(plt.gcf())

        # === Splitting ===
        X_train, X_test, y_train, y_test = train_test_split(df['clean'], df['label'], test_size=0.2, random_state=42)

        # === TF-IDF Vectorization ===
        tfidf = TfidfVectorizer()
        X_train_vec = tfidf.fit_transform(X_train)
        X_test_vec = tfidf.transform(X_test)

        # === Model Training ===
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
        report = classification_report(y_test, y_pred, output_dict=True, target_names=["Negatif", "Positif"])
        st.dataframe(pd.DataFrame(report).transpose())

        # === Prediksi Kalimat Baru ===
        st.subheader("📝 Prediksi Kalimat Baru")
        kalimat_baru = st.text_area("Masukkan kalimat ulasan:")
        if st.button("Prediksi"):
            bersih = preprocessing(kalimat_baru)
            vec = tfidf.transform([bersih])
            prediksi = model.predict(vec)[0]
            label = "Positif ✅" if prediksi == 1 else "Negatif ❌"
            st.success(f"Sentimen: **{label}**")
