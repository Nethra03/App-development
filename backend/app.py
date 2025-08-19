from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
CORS(app)

MODEL_PATH = "backend/indian_food_model.h5"
model = load_model(MODEL_PATH)

class_names = ["samosa", "dosa", "idli", "biryani", "pav_bhaji"]

def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filepath = os.path.join("backend", file.filename)
    file.save(filepath)

    img_array = prepare_image(filepath)
    preds = model.predict(img_array)
    class_idx = np.argmax(preds[0])
    food_name = class_names[class_idx]
    confidence = float(np.max(preds[0]))

    os.remove(filepath)
    return jsonify({"prediction": food_name, "confidence": confidence})

if __name__ == "__main__":
    app.run(debug=True)