# Sentiment Analysis Project

## 📌 Overview

This project is a **Sentiment Analysis Web Application** that predicts whether a given text expresses a **positive, negative, or neutral** sentiment. It uses **Machine Learning** techniques with a pre-trained model and provides a simple **web interface** built with Flask.

## 🚀 Features

* Pre-trained **Sentiment Analysis Model** (`sentiment_model.pkl`)
* **TF-IDF Vectorizer** for text feature extraction
* Web interface using **Flask + HTML (Jinja templates)**
* User can input text and get real-time predictions

## 🗂 Project Structure

```
├── chechi/
│   ├── app.py                  # Flask web app
│   ├── model.py                # Model training/loading script
│   ├── sentiment_model.pkl     # Pre-trained model
│   ├── tfidf_vectorizer.pkl    # TF-IDF vectorizer
│   ├── templates/
│   │   └── index.html          # Web UI
```

## ⚙️ Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-username/sentiment-analysis.git
   cd sentiment-analysis/main
   ```

2. **Create a virtual environment (optional but recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Linux/Mac
   venv\Scripts\activate      # On Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

*(If `requirements.txt` is missing, include Flask, scikit-learn, and any other dependencies manually.)*

## ▶️ Usage

Run the Flask app:

```bash
python app.py
```

Open the application in your browser at:

```
http://127.0.0.1:5000/
```

## 📊 Technologies Used

* **Python**
* **Flask** (Web Framework)
* **Scikit-learn** (Machine Learning)
* **Pickle** (Model Persistence)
* **HTML/CSS (Jinja Templates)**

## 🔮 Future Improvements

* Improve model accuracy using deep learning (LSTMs / Transformers)
* Add more granular sentiment categories
* Deploy on **Heroku / Render / AWS** for live access

---

✨ Feel free to fork this repo, raise issues, and contribute!

---

