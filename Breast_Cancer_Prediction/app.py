from flask import Flask, render_template, request
import numpy as np
import joblib
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "model", "breast_cancer_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "model", "scaler.pkl"))

# # Load saved model and scaler
# model = joblib.load(r"model\breast_cancer_model.pkl")
# scaler = joblib.load(r"model\scaler.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        # Get form values
        radius = float(request.form["radius"])
        texture = float(request.form["texture"])
        perimeter = float(request.form["perimeter"])
        area = float(request.form["area"])
        smoothness = float(request.form["smoothness"])

        # Prepare input for model
        features = np.array([[radius, texture, perimeter, area, smoothness]])
        features_scaled = scaler.transform(features)

        # Predict
        result = model.predict(features_scaled)[0]

        prediction = "Malignant" if result == 1 else "Benign"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
