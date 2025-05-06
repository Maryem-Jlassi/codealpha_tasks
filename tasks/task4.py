import os
from flask import Flask, request, jsonify, render_template, send_from_directory
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"  # Dossier pour les images téléchargées
app.config["RESULTS_FOLDER"] = "results"  # Dossier pour les résultats
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # Taille maximale de fichier : 16 Mo

model = YOLO("model/card_detector.pt") 

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["RESULTS_FOLDER"], exist_ok=True)


@app.route("/upload", methods=["POST"])
def upload_file():
    if "image" not in request.files:
        return jsonify({"error": "Aucune image téléchargée"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Nom de fichier vide"}), 400

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    img = cv2.imread(file_path)
    if img is None:
        return jsonify({"error": "Impossible de lire l'image"}), 400

    results = model(img)

    annotated_img = results[0].plot()  

    result_path = os.path.join(app.config["RESULTS_FOLDER"], file.filename)
    cv2.imwrite(result_path, annotated_img)

    detections = []
    for box in results[0].boxes.data.tolist():
        x1, y1, x2, y2, confidence, class_id = box
        class_name = results[0].names[int(class_id)]
        detections.append({
            "class": class_name,
            "confidence": round(confidence, 2),
            "box": [int(x1), int(y1), int(x2), int(y2)]
        })

    return jsonify({
        "detections": detections,
        "annotated_image": f"/results/{file.filename}"
    })

@app.route("/results/<filename>")
def get_result(filename):
    return send_from_directory(app.config["RESULTS_FOLDER"], filename)

if __name__ == "__main__":
    app.run(debug=True)