from flask import Flask, render_template, request, url_for
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from joblib import load
from nltk.tokenize import RegexpTokenizer


app = Flask(__name__)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    A naive bayes model that returns predictions for ham or spam.
    based on comments user inputs.
    """
    # loading our dataset
    data_path = "/home/paul/workspace/data-science/my-projects/SMS-Spam-Classifier-App-master/ham-or-spam-nb-classifier/data/processed/"  # noqa
    df = pd.read_csv(
        data_path + "spam_processed.csv", delimiter=",", encoding="latin-1"
    )

    # tokinize message just in case we missed some numbers or other chars
    token = RegexpTokenizer(r"[a-zA-Z0-9]+")
    # generate document term matrix by using scikit-learn's CountVectorizer()
    # Return a function that splits a string into a sequence of tokens
    cv = CountVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 1),
        tokenizer=token.tokenize,
    )
    X = cv.fit_transform(df["message"])
    y = df["label"]
    # loading our saved naive bayes model
    joblib_path = "/home/paul/workspace/data-science/my-projects/SMS-Spam-Classifier-App-master/ham-or-spam-nb-classifier/models/"  # noqa
    naive_bayes_model = open(joblib_path + "model.joblib", "rb")
    clf = load(naive_bayes_model)

    # return our prediction
    if request.method == "POST":
        comment = request.form["comment"]
        data = [comment]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
    return render_template("predictions.html", prediction=my_prediction)


if __name__ == "__main__":
    app.run(debug=False)
