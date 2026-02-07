from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load trained ML pipeline
with open("HealthExpense.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/prediction")
def prediction():
    return render_template("prediction.html")

@app.route("/result", methods=["POST"])
def result():

    # Collect input data
    input_df = pd.DataFrame([{
        "age": int(request.form["age"]),
        "sex": request.form["sex"],
        "bmi": float(request.form["bmi"]),
        "children": int(request.form["children"]),
        "smoker": request.form["smoker"],
        "region": request.form["region"]
    }])

    # Prediction
    prediction = model.predict(input_df)[0]

    return render_template(
        "result.html",
        prediction=round(prediction, 2),
        age=request.form["age"],
        sex=request.form["sex"],
        bmi=request.form["bmi"],
        children=request.form["children"],
        smoker=request.form["smoker"],
        region=request.form["region"]
    )

if __name__ == "__main__":
    app.run(debug=True)
