
import gradio as gr
import joblib
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load model
svm_model = joblib.load('model/svm_model.pkl')
tfidf_vectorizer = joblib.load('model/tfidf_vectorizer.pkl')
mlb = joblib.load('model/mlb.pkl')

# Preprocessing
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return tokens

# Prediction
def predict_genre_english(description):
    try:
        if not description.strip():
            return "❌ Input tidak boleh kosong."

        clean_tokens = preprocess_text(description)
        clean_text = ' '.join(clean_tokens)

        vec = tfidf_vectorizer.transform([clean_text])
        pred = svm_model.predict(vec)

        genres = mlb.inverse_transform(pred)

        if genres and len(genres[0]) > 0:
            hasil_genre = ", ".join(genres[0])
        else:
            hasil_genre = "Genre tidak terdeteksi"

        return f"✅ Success!\n\nPredicted Genre: {hasil_genre}"

    except Exception as e:
        return f"❌ Error: {str(e)}"

# UI
demo = gr.Interface(
    fn=predict_genre_english,
    inputs=gr.Textbox(
        label="Input Movie Description (English)",
        placeholder="Example: A brave warrior fights for justice in a fantasy world...",
        lines=4
    ),
    outputs=gr.Textbox(label="Prediction Result", lines=5),
    title="🎬 Movie Genre Predictor",
    description="Prediksi genre film dari deskripsi menggunakan NLP",
    theme="soft"
)

demo.launch()
