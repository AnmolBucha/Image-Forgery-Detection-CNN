import os
import secrets
from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from config import Config
from utils.image_processor import ImageProcessor
from models.model_loader import ModelLoader
from database import db

app = Flask(__name__)
app.config.from_object(Config)
os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
os.makedirs(Config.RESULT_FOLDER, exist_ok=True)

processor = ImageProcessor()


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in Config.ALLOWED_EXTENSIONS
    )


def generate_filename():
    return secrets.token_hex(16)


@app.route("/")
def home():
    models_ready = ModelLoader.is_models_available()
    stats = db.get_overall_stats()
    return render_template("index.html", models_ready=models_ready, stats=stats)


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/dashboard")
def dashboard():
    stats = db.get_overall_stats()
    daily_stats = db.get_daily_stats(7)
    return render_template("dashboard.html", stats=stats, daily_stats=daily_stats)


@app.route("/history")
def history():
    page = request.args.get("page", 1, type=int)
    per_page = 20
    offset = (page - 1) * per_page
    history_items = db.get_history(limit=per_page, offset=offset)
    total_items = db.get_history_count()
    total_pages = (total_items + per_page - 1) // per_page
    return render_template(
        "history.html",
        history=history_items,
        page=page,
        total_pages=total_pages,
        total_items=total_items,
    )


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        unique_filename = f"{generate_filename()}_{filename}"
        filepath = os.path.join(Config.UPLOAD_FOLDER, unique_filename)
        file.save(filepath)

        try:
            result_filename = f"result_{unique_filename}"
            result_filepath = os.path.join(Config.RESULT_FOLDER, result_filename)

            prediction, confidence, num_patches = processor.predict(filepath)

            if prediction is None:
                return jsonify(
                    {
                        "error": "Models not loaded. Please ensure CNN and SVM models are trained and placed in the models folder.",
                        "models_ready": False,
                    }
                ), 500

            pred_label = "Forged" if prediction == 1 else "Authentic"
            db.add_analysis(filename, pred_label, confidence, num_patches)

            result = {
                "filename": unique_filename,
                "prediction": pred_label,
                "confidence": round(confidence, 2),
                "num_patches": num_patches,
                "result_image": result_filename,
                "models_ready": True,
            }

            cv2.imwrite(result_filepath, cv2.imread(filepath))

            return jsonify(result)

        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify(
            {"error": "Invalid file type. Please upload an image file."}
        ), 400


@app.route("/batch-upload", methods=["POST"])
def batch_upload():
    if "files" not in request.files:
        return jsonify({"error": "No files part"}), 400

    files = request.files.getlist("files")
    if not files or all(f.filename == "" for f in files):
        return jsonify({"error": "No selected files"}), 400

    results = []

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            unique_filename = f"{generate_filename()}_{filename}"
            filepath = os.path.join(Config.UPLOAD_FOLDER, unique_filename)
            file.save(filepath)

            try:
                prediction, confidence, num_patches = processor.predict(filepath)

                if prediction is not None:
                    pred_label = "Forged" if prediction == 1 else "Authentic"
                    db.add_analysis(filename, pred_label, confidence, num_patches)
                    results.append(
                        {
                            "filename": filename,
                            "unique_filename": unique_filename,
                            "prediction": pred_label,
                            "confidence": round(confidence, 2),
                            "num_patches": num_patches,
                        }
                    )
                else:
                    results.append(
                        {"filename": filename, "error": "Model prediction failed"}
                    )
            except Exception as e:
                results.append({"filename": filename, "error": str(e)})

    return jsonify({"results": results})


@app.route("/api/check-models")
def check_models():
    models_ready = ModelLoader.is_models_available()
    return jsonify({"models_ready": models_ready})


@app.route("/api/stats")
def get_stats():
    stats = db.get_overall_stats()
    daily = db.get_daily_stats(7)
    return jsonify({"stats": stats, "daily": daily})


@app.route("/api/history")
def get_history_api():
    page = request.args.get("page", 1, type=int)
    per_page = 20
    offset = (page - 1) * per_page
    history_items = db.get_history(limit=per_page, offset=offset)
    return jsonify({"history": history_items, "page": page})


@app.route("/api/history/<int:item_id>", methods=["DELETE"])
def delete_history_item(item_id):
    db.delete_history_item(item_id)
    return jsonify({"success": True})


@app.route("/api/history/clear", methods=["POST"])
def clear_history():
    db.clear_history()
    return jsonify({"success": True})


@app.route("/api/train-models")
def train_models():
    return jsonify(
        {
            "message": "To train models, run the training scripts in the src/ directory first.",
            "instructions": [
                "1. First, run patch extraction: python src/extract_patches.py",
                "2. Train CNN: python src/train_net.py",
                "3. Extract features: python src/feature_extraction.py",
                "4. Train SVM: python src/svm_classification.py",
                "5. Copy models to web_app/models/ folder",
            ],
        }
    )


if __name__ == "__main__":
    import os

    port = int(os.environ.get("PORT", 5001))
    app.run(debug=True, host="0.0.0.0", port=port)
