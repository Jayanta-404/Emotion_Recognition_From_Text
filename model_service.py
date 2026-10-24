import json
import os
import time
import re
import numpy as np
import joblib
from scipy.special import softmax

# Suppress TensorFlow logging noise
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import keras

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Emotion metadata with colors, emojis, descriptions, and sentiment valence
EMOTION_META = {
    "anger": {
        "emoji": "😡",
        "color": "#ef4444",
        "gradient": "linear-gradient(135deg, #ef4444, #b91c1c)",
        "light_color": "rgba(239, 68, 68, 0.15)",
        "border_color": "rgba(239, 68, 68, 0.4)",
        "valence": -0.8,
        "arousal": 0.9,
        "description": "Feeling of displeasure, hostility, or antagonism"
    },
    "fear": {
        "emoji": "😨",
        "color": "#a855f7",
        "gradient": "linear-gradient(135deg, #a855f7, #7e22ce)",
        "light_color": "rgba(168, 85, 247, 0.15)",
        "border_color": "rgba(168, 85, 247, 0.4)",
        "valence": -0.7,
        "arousal": 0.8,
        "description": "Feeling induced by perceived danger, threat, or apprehension"
    },
    "joy": {
        "emoji": "😊",
        "color": "#10b981",
        "gradient": "linear-gradient(135deg, #10b981, #059669)",
        "light_color": "rgba(16, 185, 129, 0.15)",
        "border_color": "rgba(16, 185, 129, 0.4)",
        "valence": 0.9,
        "arousal": 0.7,
        "description": "Feeling of great happiness, pleasure, and contentment"
    },
    "love": {
        "emoji": "❤️",
        "color": "#ec4899",
        "gradient": "linear-gradient(135deg, #ec4899, #be185d)",
        "light_color": "rgba(236, 72, 153, 0.15)",
        "border_color": "rgba(236, 72, 153, 0.4)",
        "valence": 0.95,
        "arousal": 0.6,
        "description": "Intense feeling of deep affection, tenderness, or devotion"
    },
    "sadness": {
        "emoji": "😢",
        "color": "#3b82f6",
        "gradient": "linear-gradient(135deg, #3b82f6, #1d4ed8)",
        "light_color": "rgba(59, 130, 246, 0.15)",
        "border_color": "rgba(59, 130, 246, 0.4)",
        "valence": -0.9,
        "arousal": 0.3,
        "description": "Feeling of sorrow, unhappiness, heartbreak, or grief"
    },
    "surprise": {
        "emoji": "😲",
        "color": "#06b6d4",
        "gradient": "linear-gradient(135deg, #06b6d4, #0e7490)",
        "light_color": "rgba(6, 182, 212, 0.15)",
        "border_color": "rgba(6, 182, 212, 0.4)",
        "valence": 0.3,
        "arousal": 0.95,
        "description": "Feeling caused by something unexpected, astonishing, or sudden"
    }
}

class EmotionModelService:
    def __init__(self):
        self.labels = []
        self.keras_model = None
        self.tokenizer = None
        self.tfidf = None
        self.logistic_model = None
        self.svm_model = None
        self.loaded = False
        self.load_models()

    def load_models(self):
        print("Loading emotion recognition model files...")
        start_time = time.time()

        # 1. Labels
        labels_path = os.path.join(BASE_DIR, "labels.json")
        with open(labels_path, "r", encoding="utf-8") as f:
            self.labels = json.load(f)
        print(f"Loaded {len(self.labels)} labels: {self.labels}")

        # 2. Tokenizer
        tokenizer_path = os.path.join(BASE_DIR, "tokenizer.json")
        with open(tokenizer_path, "r", encoding="utf-8") as f:
            self.tokenizer = tokenizer_from_json(f.read())
        print(f"Loaded Tokenizer (vocab size: {self.tokenizer.num_words})")

        # 3. Keras Model
        keras_path = os.path.join(BASE_DIR, "emotion_model.keras")
        self.keras_model = keras.models.load_model(keras_path)
        print("Loaded Deep Learning Keras model (FastText + Conv1D + BiLSTM)")

        # 4. TF-IDF Vectorizer
        tfidf_path = os.path.join(BASE_DIR, "tfidf.joblib")
        self.tfidf = joblib.load(tfidf_path)
        print("Loaded TF-IDF Vectorizer")

        # 5. Logistic Regression
        lr_path = os.path.join(BASE_DIR, "logistic_regression.joblib")
        self.logistic_model = joblib.load(lr_path)
        print("Loaded Logistic Regression model")

        # 6. Linear SVM
        svm_path = os.path.join(BASE_DIR, "linear_svm.joblib")
        self.svm_model = joblib.load(svm_path)
        print("Loaded Linear SVM model")

        # Warm-up inference
        self._warmup()
        self.loaded = True
        print(f"All models loaded successfully in {time.time() - start_time:.2f}s")

    def _warmup(self):
        warmup_text = ["i feel happy today"]
        seqs = self.tokenizer.texts_to_sequences(warmup_text)
        padded = pad_sequences(seqs, maxlen=60, padding="post", truncating="post")
        self.keras_model.predict(padded, verbose=0)
        X = self.tfidf.transform(warmup_text)
        self.logistic_model.predict_proba(X)
        self.svm_model.decision_function(X)

    def _format_prediction_result(self, probabilities, model_name, inference_time_ms, text):
        probabilities = [float(p) for p in probabilities]
        max_idx = int(np.argmax(probabilities))
        dominant_emotion = self.labels[max_idx]
        confidence = probabilities[max_idx]

        # Emotion breakdown list sorted descending
        breakdown = []
        for i, emotion in enumerate(self.labels):
            prob = probabilities[i]
            meta = EMOTION_META.get(emotion, {})
            breakdown.append({
                "emotion": emotion,
                "probability": round(prob, 4),
                "percentage": round(prob * 100, 2),
                "emoji": meta.get("emoji", "🎭"),
                "color": meta.get("color", "#6366f1"),
                "gradient": meta.get("gradient", ""),
                "description": meta.get("description", "")
            })
        breakdown.sort(key=lambda x: x["probability"], reverse=True)

        # Overall sentiment valence (-1 to +1) and arousal
        valence = sum(EMOTION_META[e]["valence"] * p for e, p in zip(self.labels, probabilities))
        arousal = sum(EMOTION_META[e]["arousal"] * p for e, p in zip(self.labels, probabilities))
        
        sentiment = "Neutral"
        if valence > 0.2:
            sentiment = "Positive"
        elif valence < -0.2:
            sentiment = "Negative"

        # Word-level attribution highlights
        word_highlights = self._extract_word_impact(text, dominant_emotion)

        return {
            "dominant_emotion": dominant_emotion,
            "confidence": round(confidence, 4),
            "confidence_pct": round(confidence * 100, 2),
            "emoji": EMOTION_META[dominant_emotion]["emoji"],
            "color": EMOTION_META[dominant_emotion]["color"],
            "gradient": EMOTION_META[dominant_emotion]["gradient"],
            "description": EMOTION_META[dominant_emotion]["description"],
            "sentiment": sentiment,
            "valence_score": round(valence, 3),
            "arousal_score": round(arousal, 3),
            "breakdown": breakdown,
            "model_used": model_name,
            "latency_ms": round(inference_time_ms, 2),
            "word_highlights": word_highlights
        }

    def _extract_word_impact(self, text, dominant_emotion):
        """Highlights words that correlate with the emotion using TF-IDF weights."""
        words = re.findall(r"\b\w+\b", text.lower())
        if not words:
            return []

        # Find words known in tokenizer
        word_index = getattr(self.tokenizer, "word_index", {})
        highlights = []
        
        # Check against tfidf vocabulary
        tfidf_vocab = getattr(self.tfidf, "vocabulary_", {})
        
        for w in words:
            in_vocab = w in word_index or w in tfidf_vocab
            highlights.append({
                "word": w,
                "in_vocab": in_vocab
            })
        return highlights

    def predict_keras(self, text: str):
        t0 = time.perf_counter()
        seqs = self.tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seqs, maxlen=60, padding="post", truncating="post")
        preds = self.keras_model.predict(padded, verbose=0)[0]
        latency = (time.perf_counter() - t0) * 1000
        return self._format_prediction_result(preds, "Deep Learning (FastText + Conv1D + BiLSTM)", latency, text)

    def predict_logistic(self, text: str):
        t0 = time.perf_counter()
        X = self.tfidf.transform([text])
        probs = self.logistic_model.predict_proba(X)[0]
        latency = (time.perf_counter() - t0) * 1000
        return self._format_prediction_result(probs, "Logistic Regression (TF-IDF)", latency, text)

    def predict_svm(self, text: str):
        t0 = time.perf_counter()
        X = self.tfidf.transform([text])
        decision = self.svm_model.decision_function(X)[0]
        probs = softmax(decision)
        latency = (time.perf_counter() - t0) * 1000
        return self._format_prediction_result(probs, "Linear SVM (TF-IDF)", latency, text)

    def predict(self, text: str, model_type: str = "keras"):
        if model_type == "logistic":
            return self.predict_logistic(text)
        elif model_type == "svm":
            return self.predict_svm(text)
        elif model_type == "compare_all":
            return self.predict_all(text)
        else:
            return self.predict_keras(text)

    def predict_all(self, text: str):
        t0 = time.perf_counter()
        keras_res = self.predict_keras(text)
        lr_res = self.predict_logistic(text)
        svm_res = self.predict_svm(text)
        total_latency = (time.perf_counter() - t0) * 1000

        # Consensus check
        dominant_list = [keras_res["dominant_emotion"], lr_res["dominant_emotion"], svm_res["dominant_emotion"]]
        from collections import Counter
        counts = Counter(dominant_list)
        consensus_emotion, agreement_count = counts.most_common(1)[0]
        agreement_level = f"{agreement_count}/3 models agree"

        return {
            "is_comparison": True,
            "text": text,
            "consensus_emotion": consensus_emotion,
            "consensus_emoji": EMOTION_META[consensus_emotion]["emoji"],
            "consensus_color": EMOTION_META[consensus_emotion]["color"],
            "agreement_level": agreement_level,
            "total_latency_ms": round(total_latency, 2),
            "models": {
                "keras": keras_res,
                "logistic": lr_res,
                "svm": svm_res
            }
        }

    def batch_predict(self, texts: list, model_type: str = "keras"):
        results = []
        emotion_counts = {emotion: 0 for emotion in self.labels}
        start_time = time.perf_counter()

        for text in texts:
            clean_text = text.strip()
            if not clean_text:
                continue
            pred = self.predict(clean_text, model_type)
            # If compare_all was chosen for batch, pick consensus
            dom = pred.get("consensus_emotion") if pred.get("is_comparison") else pred.get("dominant_emotion")
            if dom in emotion_counts:
                emotion_counts[dom] += 1
            results.append({
                "text": clean_text,
                "prediction": pred
            })

        total_time = (time.perf_counter() - start_time) * 1000
        total_count = len(results)
        distribution = []
        for e in self.labels:
            cnt = emotion_counts[e]
            pct = round((cnt / total_count * 100), 1) if total_count > 0 else 0
            distribution.append({
                "emotion": e,
                "count": cnt,
                "percentage": pct,
                "emoji": EMOTION_META[e]["emoji"],
                "color": EMOTION_META[e]["color"]
            })

        return {
            "total_items": total_count,
            "total_time_ms": round(total_time, 2),
            "avg_latency_ms": round(total_time / max(1, total_count), 2),
            "distribution": distribution,
            "results": results
        }

    def get_system_metadata(self):
        return {
            "supported_emotions": self.labels,
            "emotion_metadata": EMOTION_META,
            "models": [
                {
                    "id": "keras",
                    "name": "Keras Neural Network (FastText + Conv1D + BiLSTM)",
                    "type": "Deep Learning (NLP)",
                    "file": "emotion_model.keras",
                    "file_size": "19.1 MB",
                    "sequence_length": 60,
                    "embedding_dim": 100,
                    "vocab_size": 30000,
                    "recommended": True,
                    "description": "Dual-branch neural network combining 1D Convolution for local n-gram feature extraction and Bidirectional LSTM for sequential context, trained on FastText word embeddings."
                },
                {
                    "id": "logistic",
                    "name": "Multinomial Logistic Regression",
                    "type": "Classical Machine Learning",
                    "file": "logistic_regression.joblib",
                    "file_size": "1.4 MB",
                    "features": "TF-IDF (30,000 max features)",
                    "recommended": False,
                    "description": "Linear probabilistic classifier trained on sublinear term-frequency inverse-document frequency representations."
                },
                {
                    "id": "svm",
                    "name": "Linear Support Vector Machine (LinearSVC)",
                    "type": "Classical Machine Learning",
                    "file": "linear_svm.joblib",
                    "file_size": "1.4 MB",
                    "features": "TF-IDF (30,000 max features)",
                    "recommended": False,
                    "description": "Max-margin hyperplane classifier optimized for high-dimensional sparse text vectors, with softmax probability calibration."
                }
            ],
            "input_type": "text",
            "input_max_tokens": 60,
            "status": "online"
        }

# Global singleton
service = None

def get_service():
    global service
    if service is None:
        service = EmotionModelService()
    return service
