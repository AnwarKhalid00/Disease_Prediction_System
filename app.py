from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model and columns
model = pickle.load(open("model/disease_model.pkl", "rb"))
columns = pickle.load(open("model/columns.pkl", "rb"))


# ================= HOME PAGE =================
@app.route("/")
def home():
    return render_template("home.html", active="home")


# ================= PREDICTION PAGE =================
@app.route("/predict_page")
def predict_page():
    return render_template("index.html", symptoms=columns, active="predict")


# ================= ABOUT PAGE =================
@app.route("/about")
def about():
    return render_template("about.html", active="about")


# ================= CONTACT PAGE =================
@app.route("/contact")
def contact():
    return render_template("contact.html", active="contact")


# ================= PREDICT FUNCTION =================
@app.route("/predict", methods=["POST"])
def predict():

    selected_symptoms = request.form.getlist("symptoms")

    # Create input vector
    input_data = [0] * len(columns)

    # Set symptom values
    for symptom in selected_symptoms:
        if symptom in columns:
            index = list(columns).index(symptom)
            input_data[index] = 1


    # ================= ADD LIFESTYLE FEATURES =================

    lifestyle_features = ["smoking", "alcohol", "exercise_level", "water_intake", "bmi"]

    for feature in lifestyle_features:
        if feature in columns:
            value = request.form.get(feature)

            if value is not None and value != "":
                index = list(columns).index(feature)
                input_data[index] = float(value)


    # Convert to numpy array
    input_data = np.array(input_data).reshape(1, -1)

    # Predict disease
    prediction = model.predict(input_data)[0]


    return render_template(
        "result.html",
        disease=prediction,
        active="predict"
    )


if __name__ == "__main__":
    app.run(debug=True)