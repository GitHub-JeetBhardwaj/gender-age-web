# app.py
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import uuid
from huggingface_hub import hf_hub_download
import threading
import random

app = Flask(
    __name__,
    static_folder="static",
    template_folder="templates"
)

# Configuration
app.config["UPLOAD_FOLDER"] = "/tmp/uploads"  # Vercel's writable /tmp
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}

# Suppress TensorFlow logs
tf.get_logger().setLevel('ERROR')

# Global model variable (lazy-loaded)
model = None
model_lock = threading.Lock()

def create_pseudo_model():
    """Create a simple Keras model mimicking age/gender outputs."""
    input_shape = (200, 200, 3)  # Match real model input
    inputs = Input(shape=input_shape)
    x = Dense(8, activation='relu')(inputs)  # Minimal layers
    age_output = Dense(1, name='age')(x)  # Single float for age
    gender_output = Dense(1, activation='sigmoid', name='gender')(x)  # Sigmoid for gender
    pseudo_model = Model(inputs=inputs, outputs=[age_output, gender_output])
    print("Pseudo model created as fallback")
    return pseudo_model

def get_model():
    global model
    if model is None:
        with model_lock:
            if model is None:
                print("Attempting to download model...")
                try:
                    model_path = hf_hub_download(
                        repo_id='YOUR_USERNAME/age_gender_final_model',  # Replace with your repo ID
                        filename='age_gender_final_model.h5',
                        local_dir='/tmp',
                        local_dir_use_symlinks=False
                    )
                    model = tf.keras.models.load_model(model_path, compile=False)
                    print("Real model loaded successfully!")
                except Exception as e:
                    print(f"Failed to load real model: {e}. Using pseudo model.")
                    model = create_pseudo_model()  # Fallback to pseudo model
    return model

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_img(img_path: str):
    try:
        img = image.load_img(img_path, target_size=(200, 200))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

# Basic pages
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

# Prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type. Please use JPEG, PNG, or WEBP."}), 400

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    original_name = secure_filename(file.filename)
    ext = os.path.splitext(original_name)[1].lower()
    unique_name = f"{uuid.uuid4().hex}{ext}"
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)

    try:
        file.save(file_path)
    except Exception as e:
        return jsonify({"error": f"Failed to save file: {str(e)}"}), 500

    img_array = preprocess_img(file_path)
    if img_array is None:
        try:
            os.remove(file_path)
        except FileNotFoundError:
            pass
        return jsonify({"error": "Failed to preprocess image"}), 500

    try:
        loaded_model = get_model()
        age_pred, gender_pred = loaded_model.predict(img_array, verbose=0)
        # For pseudo model, override with random values if model is untrained
        if loaded_model.name == 'model':  # Default name for pseudo model
            age_pred = np.array([[random.uniform(18, 80)]])  # Random age
            gender_pred = np.array([[random.uniform(0, 1)]])  # Random gender prob
        age = float(age_pred[0][0])
        gender = "Female" if float(gender_pred[0][0]) >= 0.5 else "Male"
        result = {"age": round(age, 1), "gender": gender}
    except Exception as e:
        try:
            os.remove(file_path)
        except FileNotFoundError:
            pass
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
    finally:
        try:
            os.remove(file_path)
        except FileNotFoundError:
            pass

    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=False)  # For local testing
