from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import requests
import random

app = Flask(__name__, static_folder="static", template_folder="templates")
HF_API_URL = "https://jeetbhardwaj07-age-gender-predictor-web.hf.space/predict"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded", "is_pseudo": False}), 400
    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"error": "Empty filename", "is_pseudo": False}), 400
    if not ('.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in {"jpg", "jpeg", "png", "webp"}):
        return jsonify({"error": "Unsupported file type", "is_pseudo": False}), 400

    try:
        response = requests.post(HF_API_URL, files={"file": (file.filename, file.stream, file.mimetype)})
        response.raise_for_status()
        data = response.json()
        if "error" in data:
            raise Exception(data["error"])
        return jsonify({**data, "is_pseudo": False})
    except Exception as e:
        # Pseudo fallback
        pseudo_age = round(random.uniform(18, 80), 1)
        pseudo_gender = "Female" if random.random() > 0.5 else "Male"
        return jsonify({
            "error": f"HF API failed: {str(e)}",
            "age": pseudo_age,
            "gender": pseudo_gender,
            "is_pseudo": True
        }), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=False)
