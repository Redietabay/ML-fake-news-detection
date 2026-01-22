from flask import Flask, render_template, request
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

app = Flask(__name__)

# ------------------------
# Load artifacts
# ------------------------
with open("../notebook/artifacts/tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("../notebook/artifacts/nb_model.pkl", "rb") as f:
    nb_model = pickle.load(f)

with open("../notebook/artifacts/logistic_model.pkl", "rb") as f:
    log_model = pickle.load(f)

with open("../notebook/artifacts/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

lstm_model = load_model("../notebook/artifacts/lstm_model.keras")


# ------------------------
# Helper prediction functions
# ------------------------
def nb_log_predict(text, model):
    vec = tfidf.transform([text])
    probs = model.predict_proba(vec)[0]
    label = "REAL NEWS" if probs[1] > 0.5 else "FAKE NEWS"
    return label, round(max(probs) * 100, 2)


def lstm_predict(text):
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=200)
    pred = lstm_model.predict(pad, verbose=0)[0][0]
    label = "REAL NEWS" if pred >= 0.5 else "FAKE NEWS"
    conf = pred if pred >= 0.5 else (1 - pred)
    return label, round(conf * 100, 2)


# ------------------------
# Main route
# ------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    data = {
        "result": None,
        "prob": 0,
        "news_text": "",
        "selected_model": "nb",
        "nb_prob": 0,
        "log_prob": 0,
        "lstm_prob": 0
    }

    if request.method == "POST":
        text = request.form.get("news_text", "").strip()
        model_type = request.form.get("model_type", "nb")

        if text:
            # Selected model
            if model_type == "nb":
                data["result"], data["prob"] = nb_log_predict(text, nb_model)
            elif model_type == "log":
                data["result"], data["prob"] = nb_log_predict(text, log_model)
            else:
                data["result"], data["prob"] = lstm_predict(text)

            # Cross-model confidence
            _, data["nb_prob"] = nb_log_predict(text, nb_model)
            _, data["log_prob"] = nb_log_predict(text, log_model)
            _, data["lstm_prob"] = lstm_predict(text)

            data["news_text"] = text
            data["selected_model"] = model_type

    return render_template("index.html", **data)


if __name__ == "__main__":
    app.run(debug=True)
