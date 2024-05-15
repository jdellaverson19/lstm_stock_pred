from flask import Flask, render_template, request, jsonify
import json
from package import lstm_preds  # Assuming this file contains your prediction logic

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/data", methods=["POST"])
def data():
    search = request.form["search"]
    predictions = lstm_preds.getPrediction(
        search
    )  # Adjust this according to your prediction logic
    return render_template("predPage.html", pred=json.dumps(predictions))


if __name__ == "__main__":
    app.run(debug=True)
