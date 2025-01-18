from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load model and scaler
model = tf.keras.models.load_model("models/KNN.h5")
scaler = joblib.load("models/scaler.pkl")

# Load label encoder
labels = ["Depresi", "Gangguan Bipolar", "Skizofrenia", "Demensia", "Gangguan Tumbuh Kembang"]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get data from form
        data = [
            float(request.form["usia"]),
            int(request.form["jenis_kelamin"]),
            int(request.form["riwayat_keluarga"]),
            int(request.form["gejala1"]),
            int(request.form["gejala2"]),
            int(request.form["gejala3"]),
            int(request.form["gejala4"]),
            int(request.form["gejala5"]),
            float(request.form["durasi_gejala"]),
        ]
        
        # Scale input data
        data_scaled = scaler.transform([data])
        
        # Predict
        prediction = model.predict(data_scaled)
        diagnosis = labels[np.argmax(prediction)]
        
        return render_template("index.html", prediction_text=f"Diagnosis: {diagnosis}")
    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
