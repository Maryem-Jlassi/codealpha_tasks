import cv2
import numpy as np
import os
import logging
from pathlib import Path
import pytesseract
import re
from datetime import datetime
import traceback

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration des patterns de validation pour chaque type de champ
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
        
        # Vérification du fichier
        if not os.path.exists(file_path):
            logger.error(f"Fichier non trouvé: {file_path}")
            return [], None, "Fichier non trouvé"

        # Vérification du modèle
        if model is None:
            logger.error("Le modèle YOLO n'est pas chargé")
            return [], None, "Modèle non chargé"
            
        logger.info(f"Modèle YOLO chargé avec {len(model.names)} classes: {model.names}")

        # Lecture de l'image
        img = cv2.imread(file_path)
        if img is None:
            logger.error(f"Impossible de lire l'image: {file_path}")
            return [], None, "Impossible de lire l'image"

        # Log des dimensions de l'image
        logger.info(f"Image dimensions: {img.shape[:2]}")
        
        # Configuration du modèle avec des paramètres optimisés par classe
        model.conf = 0.05  # Seuil de confiance très bas pour plus de détections
        model.iou = 0.2    # Seuil IOU très bas pour NMS
        logger.info(f"Model parameters - conf: {model.conf}, iou: {model.iou}")

        # Prétraitement de l'image
        processed_img = preprocess_image(img)
        
        # Sauvegarde de l'image prétraitée pour debug
        debug_path = os.path.join(results_folder, "debug_preprocessed.jpg")
        cv2.imwrite(debug_path, processed_img)
        logger.info(f"Image prétraitée sauvegardée: {debug_path}")
        
        # Détection
        results = model(processed_img)
        
        # Log des détections brutes
        logger.info(f"Total detections: {len(results[0].boxes)}")
        if len(results[0].boxes) == 0:
            logger.warning("No valid detections found in the document")
            # Essayer avec l'image originale
            logger.info("Trying detection with original image...")
            results = model(img)
            logger.info(f"Total detections with original image: {len(results[0].boxes)}")
            if len(results[0].boxes) == 0:
                return [], None, "Aucune détection trouvée dans le document"

        # Création de l'image annotée
        annotated_img = results[0].plot()
        result_filename = f"med_annotated_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.path.basename(file_path)}"
        result_path = os.path.join(results_folder, result_filename)
        cv2.imwrite(result_path, annotated_img)

        # Traitement des détections
        detections = []
        height, width = img.shape[:2]

        for box in results[0].boxes.data.tolist():
            try:
                x1, y1, x2, y2, confidence, class_id = box
                class_name = results[0].names[int(class_id)]
                
                # Calcul des coordonnées relatives
                rel_x1, rel_y1 = x1/width, y1/height
                rel_x2, rel_y2 = x2/width, y2/height
                
                # Extraction du texte
                field_img = img[int(y1):int(y2), int(x1):int(x2)]
                text = "Texte non extrait"  # Valeur par défaut
                text_confidence = 0.0
                
                # Traitement spécial pour les blocs
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

        # Tri des détections par confiance
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        logger.info(f"Valid detections: {len(detections)}")
        
        # Log des détections par classe
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
        # Sauvegarde de l'image originale pour debug
        original = img.copy()
        
        # Redimensionnement de l'image
        img = cv2.resize(img, (640, 640))
        
        # Conversion en niveaux de gris
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Amélioration du contraste
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Réduction du bruit
        denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
        
        # Seuillage adaptatif
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Dilatation pour connecter les composants de texte
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        dilated = cv2.dilate(thresh, kernel, iterations=1)
        
        # Conversion en RGB pour le modèle YOLO
        rgb = cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)
        
        return rgb
        
    except Exception as e:
        logger.error(f"Error in image preprocessing: {str(e)}")
        return original  # Retourner l'image originale en cas d'erreur

def extract_medical_text(img, field_type):
    """
    Extrait le texte d'un champ médical détecté.
    """
    try:
        # Prétraitement spécifique au champ
        processed_img = preprocess_image(img)
        
        # Configuration OCR optimisée
        config = '--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,/-()° '
        
        # Extraction du texte
        text = pytesseract.image_to_string(processed_img, config=config)
        text = text.strip()
        
        # Si aucun texte n'est trouvé, essayer avec une configuration différente
        if not text:
            config = '--psm 6 --oem 3'
            text = pytesseract.image_to_string(processed_img, config=config)
            text = text.strip()
        
        # Si toujours aucun texte, essayer avec l'image originale
        if not text:
            text = pytesseract.image_to_string(img, config=config)
            text = text.strip()
        
        # Validation du texte
        cleaned_text, is_valid = validate_field_value(field_type, text)
        
        # Ajustement de la confiance
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
        # Nettoyage du texte
        text = text.strip().lower()
        text = re.sub(r'\s+', ' ', text)
        
        # Validation selon le type de champ
        if field_type in FIELD_PATTERNS:
            pattern = FIELD_PATTERNS[field_type]
            if re.match(pattern, text):
                return text, True
            return text, False
            
        # Traitement spécial pour les noms de médicaments
        if field_type == 'medicine_name':
            if text and re.search(r'[a-zA-Z]', text):
                return text, True
            return text, False
            
        # Traitement spécial pour les diagnostics
        if field_type == 'diagnosis':
            if text and len(text) > 3:
                return text, True
            return text, False
            
        # Traitement spécial pour l'historique
        if field_type == 'history':
            if text and len(text) > 3:
                return text, True
            return text, False
            
        # Traitement spécial pour le type de médicament
        if field_type == 'medicine_type':
            if text and re.search(r'[a-zA-Z]', text):
                return text, True
            return text, False
            
        return text, True
        
    except Exception as e:
        logger.error(f"Erreur lors de la validation du champ {field_type}: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return text, False