import cv2
import numpy as np
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def process_document(file_path, model, results_folder):
    """
    Process a document image to detect ID card fields.
    
    Args:
        file_path (str): Path to the uploaded document image
        model: YOLO model for detection
        results_folder (str): Path to save results
        
    Returns:
        tuple: (detections, result_path)
    """
    try:
        img = cv2.imread(file_path)
        if img is None:
            raise ValueError("Could not read image file")
            
        height, width = img.shape[:2]
        
        logger.info(f"Running detection on image: {os.path.basename(file_path)}")
        results = model(img)
        
        annotated_img = results[0].plot()
        
        result_filename = f"annotated_{os.path.basename(file_path)}"
        result_path = os.path.join(results_folder, result_filename)
        
        cv2.imwrite(result_path, annotated_img)
        
        detections = []
        for box in results[0].boxes.data.tolist():
            x1, y1, x2, y2, confidence, class_id = box
            class_name = results[0].names[int(class_id)]
            
            rel_x1, rel_y1 = x1/width, y1/height
            rel_x2, rel_y2 = x2/width, y2/height
            
            detections.append({
                "class": class_name,
                "confidence": round(confidence, 2),
                "box": [int(x1), int(y1), int(x2), int(y2)],
                "rel_box": [float(rel_x1), float(rel_y1), float(rel_x2), float(rel_y2)]
            })
        
        logger.info(f"Detected {len(detections)} fields in document")
        return detections, result_path
        
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise

def extract_field_content(img, detection):
    """
    Extract the content from a detected field using OCR.
    This is a placeholder for potential OCR integration.
    
    Args:
        img: Image array
        detection: Detection data with bounding box
        
    Returns:
        str: Extracted text
    """
   
    x1, y1, x2, y2 = detection["box"]
    field_img = img[y1:y2, x1:x2]
    
   
    return "OCR text extraction would go here"