from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import cv2
from ultralytics import YOLO
import ollama
from tasks.task1 import translate_text, process_voice_input
from tasks.task2 import load_all_documents, create_faiss_index, retrieve_relevant_chunks, build_rag_prompt

app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = "uploads"  
app.config["RESULTS_FOLDER"] = "results"  
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["RESULTS_FOLDER"], exist_ok=True)

model = YOLO("model/card_detector.pt") 

print("Chargement de la base de connaissances pour le chatbot...")
all_chunks = load_all_documents()
if not all_chunks:
    raise ValueError("Aucune donnée disponible dans la base de connaissances.")
index, chunks = create_faiss_index(all_chunks)
print(f"Base de connaissances chargée avec succès ! {len(all_chunks)} segments disponibles.")

@app.route("/")
def home():
    countries = {
        "am-ET": "Amharic", "ar-SA": "Arabic", "be-BY": "Bielarus", "bem-ZM": "Bemba", "bi-VU": "Bislama",
        "bn-IN": "Bengali", "bo-CN": "Tibetan", "br-FR": "Breton", "bs-BA": "Bosnian", "ca-ES": "Catalan",
        "cop-EG": "Coptic", "cs-CZ": "Czech", "cy-GB": "Welsh", "da-DK": "Danish", "dz-BT": "Dzongkha",
        "de-DE": "German", "dv-MV": "Maldivian", "el-GR": "Greek", "en-GB": "English", "es-ES": "Spanish",
        "et-EE": "Estonian", "eu-ES": "Basque", "fa-IR": "Persian", "fi-FI": "Finnish", "fn-FNG": "Fanagalo",
        "fo-FO": "Faroese", "fr-FR": "French", "gl-ES": "Galician", "gu-IN": "Gujarati", "ha-NE": "Hausa",
        "he-IL": "Hebrew", "hi-IN": "Hindi", "hr-HR": "Croatian", "hu-HU": "Hungarian", "id-ID": "Indonesian",
        "is-IS": "Icelandic", "it-IT": "Italian", "ja-JP": "Japanese", "kk-KZ": "Kazakh", "km-KM": "Khmer",
        "kn-IN": "Kannada", "ko-KR": "Korean", "ku-TR": "Kurdish", "ky-KG": "Kyrgyz", "la-VA": "Latin",
        "lo-LA": "Lao", "lv-LV": "Latvian", "men-SL": "Mende", "mg-MG": "Malagasy", "mi-NZ": "Maori",
        "ms-MY": "Malay", "mt-MT": "Maltese", "my-MM": "Burmese", "ne-NP": "Nepali", "niu-NU": "Niuean",
        "nl-NL": "Dutch", "no-NO": "Norwegian", "ny-MW": "Nyanja", "ur-PK": "Pakistani", "pau-PW": "Palauan",
        "pa-IN": "Panjabi", "ps-PK": "Pashto", "pis-SB": "Pijin", "pl-PL": "Polish", "pt-PT": "Portuguese",
        "rn-BI": "Kirundi", "ro-RO": "Romanian", "ru-RU": "Russian", "sg-CF": "Sango", "si-LK": "Sinhala",
        "sk-SK": "Slovak", "sm-WS": "Samoan", "sn-ZW": "Shona", "so-SO": "Somali", "sq-AL": "Albanian",
        "sr-RS": "Serbian", "sv-SE": "Swedish", "sw-SZ": "Swahili", "ta-LK": "Tamil", "te-IN": "Telugu",
        "tet-TL": "Tetum", "tg-TJ": "Tajik", "th-TH": "Thai", "ti-TI": "Tigrinya", "tk-TM": "Turkmen",
        "tl-PH": "Tagalog", "tn-BW": "Tswana", "to-TO": "Tongan", "tr-TR": "Turkish", "uk-UA": "Ukrainian",
        "uz-UZ": "Uzbek", "vi-VN": "Vietnamese", "wo-SN": "Wolof", "xh-ZA": "Xhosa", "yi-YD": "Yiddish",
        "zu-ZA": "Zulu"
    }
    return render_template("index.html", countries=countries)

@app.route("/document_processing", methods=["GET", "POST"])
def document_processing():
    try:
        if request.method == "POST":
            if "document" not in request.files:
                return jsonify({"error": "Aucune image téléchargée."}), 400

            file = request.files["document"]
            if file.filename == "":
                return jsonify({"error": "Nom de fichier vide."}), 400

            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(file_path)

            img = cv2.imread(file_path)
            if img is None:
                return jsonify({"error": "Impossible de lire l'image."}), 400

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

        return render_template("document_processing.html")
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/results/<filename>")
def get_result(filename):
    return send_from_directory(app.config["RESULTS_FOLDER"], filename)

if __name__ == "__main__":
    app.run(debug=True)
