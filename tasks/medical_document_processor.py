import cv2
import numpy as np
import os
import logging
from pathlib import Path
import pytesseract
import re
from datetime import datetime
import traceback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FIELD_PATTERNS = {
    'age': r'^\d{1,3}(?:\s*(?:years?|yrs?|y))?$',
    'bp': r'^\d{2,3}/\d{2,3}(?:\s*(?:mmHg|mm\s*Hg))?$',
    'date': r'^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}$',
    'temp': r'^\d{2,3}(?:\.\d{1,2})?(?:\s*(?:°C|C|F|°F))?$',
    'weight': r'^\d{2,3}(?:\.\d{1,2})?(?:\s*(?:kg|g|lbs?))?$',
    'weight-': r'^\d{2,3}(?:\.\d{1,2})?(?:\s*(?:kg|g|lbs?))?$',
    'medicine_dose': r'^\d{1,3}(?:\.\d{1,2})?(?:\s*(?:mg|g|ml|cc|IU|units?))?$',
    'medicine_power': r'^\d{1,3}(?:\.\d{1,2})?(?:\s*(?:mg|g|ml|cc|IU|units?))?$',
    'gender': r'^(?:male|female|m|f|homme|femme)$',
    'name': r'^[A-Za-zÀ-ÿ\s\'-]{2,50}$',
    'ww': r'^\d{1,3}(?:\.\d{1,2})?(?:\s*(?:kg|g|lbs?))?$'
}

def process_medical_document(file_path, model, results_folder):
    """
    Traite un document médical pour détecter et extraire les informations importantes.
    
    Args:
        file_path (str): Chemin vers l'image du document
        model: Modèle YOLO pour la détection
        results_folder (str): Dossier pour sauvegarder les résultats
        
    Returns:
        tuple: (détections, chemin_du_résultat)
    """
    try:
        logger.info(f"Processing image: {os.path.basename(file_path)}")
        
        if not os.path.exists(file_path):
            logger.error(f"Fichier non trouvé: {file_path}")
            return [], None, "Fichier non trouvé"

        if model is None:
            logger.error("Le modèle YOLO n'est pas chargé")
            return [], None, "Modèle non chargé"
            
        logger.info(f"Modèle YOLO chargé avec {len(model.names)} classes: {model.names}")

        img = cv2.imread(file_path)
        if img is None:
            logger.error(f"Impossible de lire l'image: {file_path}")
            return [], None, "Impossible de lire l'image"

        logger.info(f"Image dimensions: {img.shape[:2]}")
        
        model.conf = 0.05  
        model.iou = 0.2   
        logger.info(f"Model parameters - conf: {model.conf}, iou: {model.iou}")

        processed_img = preprocess_image(img)
        
        debug_path = os.path.join(results_folder, "debug_preprocessed.jpg")
        cv2.imwrite(debug_path, processed_img)
        logger.info(f"Image prétraitée sauvegardée: {debug_path}")
        
        results = model(processed_img)
        
        logger.info(f"Total detections: {len(results[0].boxes)}")
        if len(results[0].boxes) == 0:
            logger.warning("No valid detections found in the document")
            logger.info("Trying detection with original image...")
            results = model(img)
            logger.info(f"Total detections with original image: {len(results[0].boxes)}")
            if len(results[0].boxes) == 0:
                return [], None, "Aucune détection trouvée dans le document"

        annotated_img = results[0].plot()
        result_filename = f"med_annotated_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.path.basename(file_path)}"
        result_path = os.path.join(results_folder, result_filename)
        cv2.imwrite(result_path, annotated_img)

        detections = []
        height, width = img.shape[:2]

        for box in results[0].boxes.data.tolist():
            try:
                x1, y1, x2, y2, confidence, class_id = box
                class_name = results[0].names[int(class_id)]
                
                rel_x1, rel_y1 = x1/width, y1/height
                rel_x2, rel_y2 = x2/width, y2/height
                
                field_img = img[int(y1):int(y2), int(x1):int(x2)]
                text = "Texte non extrait"  
                text_confidence = 0.0
                
                if class_name == 'block':
                    text = "Bloc d'information"
                    text_confidence = 1.0
                else:
                    try:
                        text, text_confidence = extract_medical_text(field_img, class_name)
                    except Exception as e:
                        logger.error(f"Error extracting text: {str(e)}")
                
                detections.append({
                    "class": class_name,
                    "confidence": round(float(confidence), 2),
                    "box": [int(x1), int(y1), int(x2), int(y2)],
                    "rel_box": [float(rel_x1), float(rel_y1), float(rel_x2), float(rel_y2)],
                    "text": text,
                    "text_confidence": round(float(text_confidence), 2)
                })
                
            except Exception as e:
                logger.error(f"Error processing detection: {str(e)}")
                continue

        detections.sort(key=lambda x: x['confidence'], reverse=True)
        logger.info(f"Valid detections: {len(detections)}")
        
        class_counts = {}
        for det in detections:
            class_name = det['class']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        logger.info(f"Detections by class: {class_counts}")

        return detections, result_path, None

    except Exception as e:
        logger.error(f"Error processing medical document: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return [], None, f"Error processing document: {str(e)}"

def preprocess_image(img):
    """
    Prétraite l'image pour améliorer la détection.
    """
    try:
        original = img.copy()
        
        img = cv2.resize(img, (640, 640))
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
        
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        dilated = cv2.dilate(thresh, kernel, iterations=1)
        
        rgb = cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)
        
        return rgb
        
    except Exception as e:
        logger.error(f"Error in image preprocessing: {str(e)}")
        return original 

def extract_medical_text(img, field_type):
    """
    Extrait le texte d'un champ médical détecté.
    """
    try:
        processed_img = preprocess_image(img)
        
        config = '--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,/-()° '
        
        text = pytesseract.image_to_string(processed_img, config=config)
        text = text.strip()
        
        if not text:
            config = '--psm 6 --oem 3'
            text = pytesseract.image_to_string(processed_img, config=config)
            text = text.strip()
        
        if not text:
            text = pytesseract.image_to_string(img, config=config)
            text = text.strip()
        
        cleaned_text, is_valid = validate_field_value(field_type, text)
        
        confidence = 1.0 if is_valid else 0.5
        if not text:
            confidence = 0.0
        
        return cleaned_text, confidence
        
    except Exception as e:
        logger.error(f"Error extracting text: {str(e)}")
        return "Erreur d'extraction", 0.0

def validate_field_value(field_type, text):
    """
    Valide et nettoie la valeur d'un champ selon son type.
    """
    try:
        text = text.strip().lower()
        text = re.sub(r'\s+', ' ', text)
        
        if field_type in FIELD_PATTERNS:
            pattern = FIELD_PATTERNS[field_type]
            if re.match(pattern, text):
                return text, True
            return text, False
            
        if field_type == 'medicine_name':
            if text and re.search(r'[a-zA-Z]', text):
                return text, True
            return text, False
            
        if field_type == 'diagnosis':
            if text and len(text) > 3:
                return text, True
            return text, False
            
        if field_type == 'history':
            if text and len(text) > 3:
                return text, True
            return text, False
            
        if field_type == 'medicine_type':
            if text and re.search(r'[a-zA-Z]', text):
                return text, True
            return text, False
            
        return text, True
        
    except Exception as e:
        logger.error(f"Erreur lors de la validation du champ {field_type}: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return text, False
