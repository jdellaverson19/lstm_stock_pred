from flask import Flask, request, render_template
from package import lstm_model_pred, lstm_preds

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    #   lstm_model.makeModel(1, 2)
    return render_template("index.html")


@app.route("/data", methods=["POST"])
def data():
    if request.method == "POST":
        print(request)
        symbol = request.form["search"]
        print("the symbol is " + symbol)
        pred = lstm_preds.getPrediction(symbol)
        # pred = lstm_model_pred.doItAll(5, 3, symbol)
        return render_template("predPage.html", pred=pred)


if __name__ == "__main__":
    app.run()
