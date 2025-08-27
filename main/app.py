from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# so here we are import our model which trained 
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form['text']
    
    text_tfidf = vectorizer.transform([text])
    
    # Predict sentiment
    prediction = model.predict(text_tfidf)[0]
    
    
    sentiment_label = {1: "Positive", 0: "Negative"}
    sentiment = sentiment_label.get(prediction, "Unknown")

    return render_template('index.html', text=text, sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)