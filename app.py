"""
House Price Prediction - Flask Web Application
Serves interactive prediction UI and REST API.
"""

import os
import json
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
from predict import predict_house_price, load_resources

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
IMAGES_DIR = os.path.join(BASE_DIR, "static", "images")

# Preload models
try:
    lr_model, rf_model, metadata = load_resources()
    print("[INFO] Models and metadata loaded successfully.")
except Exception as e:
    print(f"[WARNING] Could not preload models: {e}")
    metadata = {}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/metadata", methods=["GET"])
def get_metadata():
    with open(os.path.join(MODELS_DIR, "metadata.json"), "r") as f:
        data = json.load(f)
    return jsonify(data)

@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        location = data.get("location")
        total_sqft = float(data.get("total_sqft", 1000))
        bhk = int(data.get("bhk", 2))
        bathrooms = int(data.get("bathrooms", 2))
        property_age = int(data.get("property_age", 5))
        parking = int(data.get("parking", 1))
        
        if not location:
            return jsonify({"status": "error", "message": "Location is required"}), 400
            
        result = predict_house_price(
            location=location,
            total_sqft=total_sqft,
            bhk=bhk,
            bathrooms=bathrooms,
            property_age=property_age,
            parking=parking
        )
        return jsonify({"status": "success", "data": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/static/images/<path:filename>")
def serve_image(filename):
    return send_from_directory(IMAGES_DIR, filename)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
