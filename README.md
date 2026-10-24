# Emotion Recognition from Text

A web application that reads a sentence and tells you which emotion it expresses: **anger, fear, joy, love, sadness or surprise**.

It combines a deep learning model (Keras, Conv1D + BiLSTM) with two classical machine learning models (Logistic Regression and Linear SVM) behind a **FastAPI** backend, so you can run one model or compare all three on the same text.

---

## Features

- Predicts one of 6 emotions with a confidence score for every class
- Three models in one app: **Keras deep model**, **Logistic Regression**, **Linear SVM**
- **Compare mode**: runs all three models and shows whether they agree
- **Batch mode**: analyse up to 200 texts in one request, with an emotion distribution summary
- Extra outputs for each prediction: sentiment (positive / negative / neutral), valence and arousal scores, and response time in milliseconds
- Ready-made example sentences for every emotion
- Interactive API documentation at `/docs` (Swagger UI)

---

## Project pipeline

The project follows a 10-step pipeline, from choosing the data to deploying the app.

![Project pipeline](docs/pipeline.jpg)

| Stage | Steps | What happens |
|---|---|---|
| Data | 1 to 4 | Choose labels and dataset, explore it, clean the text, balance and split |
| Models | 5 to 8 | Baseline ML models, word embeddings, deep model, tuning |
| Delivery | 9 to 10 | Evaluate and fix errors, then deploy as a web app |

The ideas behind the pipeline come from four research papers on text emotion recognition (machine learning on social media text, the Vietnamese UIT-VSMEC corpus, semantic labels with mixture models, and the Semantic-Emotion Neural Network).

---

## Models

| Model | Type | Input | Details |
|---|---|---|---|
| Keras network | Deep learning | Word sequences (max 60 tokens) | FastText-style 100-d embeddings, Conv1D branch for local phrases, BiLSTM branch for sentence context, vocabulary of 30,000 words |
| Logistic Regression | Classical ML | TF-IDF (30,000 features) | Multinomial classifier with sublinear TF-IDF |
| Linear SVM | Classical ML | TF-IDF (30,000 features) | Max-margin classifier; scores are passed through a softmax to show percentages |

> The SVM percentages are normalised scores, not calibrated probabilities. Use them to compare emotions within one text, not as exact likelihoods.

---

## Dataset

The models are trained on an English short-text emotion dataset with six labels: `anger`, `fear`, `joy`, `love`, `sadness`, `surprise`.

| Split | Texts |
|---|---|
| Train | 16,000 |
| Validation | 2,000 |
| Test | 2,000 |

Things to know about this data:

- The classes are **unbalanced**: joy (about 34%) and sadness (about 29%) are much larger than surprise (about 4%). Because of this, macro-F1 is reported along with accuracy.
- The text is already lowercase with punctuation removed.
- Labels come from social media style text, so some are subjective.

---

## Results

Measured on the 2,000-text test set:

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Logistic Regression (TF-IDF) | 86.90% | 0.831 |
| Linear SVM (TF-IDF) | 88.45% | 0.835 |
| Keras Conv1D + BiLSTM | _add your score_ | _add your score_ |

> **Before you publish:** run your evaluation notebook for the Keras model and fill in its row. The two classical rows above were measured directly from the saved `.joblib` files on the test split.

---

## Project structure

```
.
├── app.py                     # FastAPI server and API routes
├── model_service.py           # Loads the models and runs predictions
├── labels.json                # Class names in model output order
├── tokenizer.json             # Keras tokenizer (word to number mapping)
├── emotion_model.keras        # Trained deep learning model
├── tfidf.joblib               # TF-IDF vectorizer
├── logistic_regression.joblib # Trained Logistic Regression
├── linear_svm.joblib          # Trained Linear SVM
├── static/                    # Frontend files (index.html, CSS, JS)
├── docs/
│   └── pipeline.jpg           # Pipeline diagram used in this README
├── run_server.bat             # One-click start on Windows
├── requirements.txt
└── README.md
```

---

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install the requirements

```bash
pip install -r requirements.txt
```

### 4. Start the server

```bash
python app.py
```

On Windows you can also double-click `run_server.bat`.

### 5. Open the app

- Web app: http://localhost:8000
- API documentation: http://localhost:8000/docs

The first start takes a few seconds because the models are loaded and warmed up once.

---

## API reference

| Method | Route | What it does |
|---|---|---|
| `GET` | `/` | Opens the web interface |
| `GET` | `/api/health` | Checks that the models are loaded |
| `GET` | `/api/metadata` | Model details, labels and emotion information |
| `GET` | `/api/presets` | Example sentences for each emotion |
| `POST` | `/api/predict` | Predict the emotion of one text |
| `POST` | `/api/batch` | Predict the emotion of many texts (up to 200) |

The `model` field accepts `keras` (default), `logistic`, `svm` or `compare_all`.

### Predict one text

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I just got the job and I cannot stop smiling", "model": "keras"}'
```

Main fields in the response:

```json
{
  "dominant_emotion": "joy",
  "confidence_pct": 97.4,
  "sentiment": "Positive",
  "valence_score": 0.81,
  "arousal_score": 0.69,
  "breakdown": [
    { "emotion": "joy", "percentage": 97.4 },
    { "emotion": "love", "percentage": 1.2 }
  ],
  "model_used": "Deep Learning (FastText + Conv1D + BiLSTM)",
  "latency_ms": 18.5
}
```

_The numbers above are only an example of the format._

### Compare all three models

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I feel so lonely tonight", "model": "compare_all"}'
```

The response includes each model's result, the `consensus_emotion`, and an `agreement_level` such as `3/3 models agree`.

### Batch prediction

```bash
curl -X POST http://localhost:8000/api/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["I am so proud of you", "This is terrifying"], "model": "logistic"}'
```

The response includes every prediction plus the share of each emotion across the batch.

---

## Tech stack

- **Backend:** Python, FastAPI, Uvicorn
- **Deep learning:** TensorFlow / Keras
- **Classical ML:** scikit-learn, joblib, SciPy
- **Frontend:** HTML, CSS and JavaScript served from `static/`

---

## Troubleshooting

- **Error when loading the `.joblib` files:** install the same scikit-learn version that was used for training. Run `pip freeze | grep scikit-learn` on the training machine and pin that version in `requirements.txt`.
- **Error when loading `emotion_model.keras`:** use TensorFlow 2.16 or newer, which includes Keras 3.
- **The page shows only a short message:** the `static/` folder with `index.html` is missing.
- **Port 8000 already in use:** change the `port` value at the bottom of `app.py`.

---

## Limitations

- Works on **English** text only
- Only six emotions; mixed feelings (for example, happy and nervous together) are reduced to the strongest one
- Sarcasm, slang and very short texts are hard for all models
- Rare emotions such as surprise have less training data, so they are predicted less reliably

## Future work

- Fine-tune a transformer model (for example DistilBERT) and compare it with the current models
- Add more emotions such as disgust, using a larger dataset
- Support more languages
- Add a Docker setup for one-command deployment

---

## Author

**Jayanta Dey**
B.Tech Computer Science Engineering, Brainware University
