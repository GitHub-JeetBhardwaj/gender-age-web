# app.py
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import uuid

app = Flask(
    __name__,
    static_folder="static",
    template_folder="templates"
)

# Configuration
app.config["UPLOAD_FOLDER"] = os.path.join(os.getcwd(), "uploads")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB hard cap
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}

# Suppress TensorFlow logs for cleaner output
tf.get_logger().setLevel('ERROR')

# Load the saved model once on startup
try:
    model = load_model('age_gender_final_model.h5', compile=False)
except Exception as e:
    print(f"Error loading model: {e}")
    raise SystemExit(1)

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_img(img_path: str):
    try:
        # Load and preprocess image to match training input shape (200, 200, 3)
        img = image.load_img(img_path, target_size=(200, 200))
        img_array = image.img_to_array(img) / 255.0  # Rescale to [0, 1]
        img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 200, 200, 3)
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

# Basic pages
@app.route("/")
def index():
    # Expects templates/index.html
    return render_template("index.html")

@app.route("/about")
def about():
    # Expects templates/about.html
    return render_template("about.html")

@app.route("/contact")
def contact():
    # Expects templates/contact.html
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

    # Ensure upload folder exists
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    # Create a safe, unique filename to avoid collisions
    original_name = secure_filename(file.filename)
    ext = os.path.splitext(original_name)[1].lower()
    unique_name = f"{uuid.uuid4().hex}{ext}"
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)

    try:
        file.save(file_path)
    except Exception as e:
        return jsonify({"error": f"Failed to save file: {str(e)}"}), 500

    # Preprocess image
    img_array = preprocess_img(file_path)
    if img_array is None:
        try:
            os.remove(file_path)
        except FileNotFoundError:
            pass
        return jsonify({"error": "Failed to preprocess image"}), 500

    # Make prediction
    try:
        age_pred, gender_pred = model.predict(img_array, verbose=0)
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
        # Clean up uploaded file regardless of outcome
        try:
            os.remove(file_path)
        except FileNotFoundError:
            pass

    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=False)
