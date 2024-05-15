from flask import Flask, render_template, request, jsonify
import json
from package import lstm_preds  # Assuming this file contains your prediction logic

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/data", methods=["POST"])
def data():
    search = request.form["search"].upper()  # Ensure stock symbol is in upper case
    predictions = lstm_preds.getPrediction(search)  # Now returns the image filename
    print(predictions[0][0])
    return render_template(
        "predPage.html",
        pred=(predictions[0][0]),
        image_path=f"static/{search}.png",
        stock_name=search,
    )


if __name__ == "__main__":
    app.run(debug=True)
