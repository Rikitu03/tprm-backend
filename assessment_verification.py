import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
import fitz
from PIL import Image
import matplotlib.pyplot as plt
import os
import re
import cv2
import numpy as np
import pytesseract
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
import pickle
import fitz
from PIL import Image
import matplotlib.pyplot as plt
from transformers import pipeline
import difflib


IMAGE_SIZE = (512, 512)
MODEL_PATH = "resnet50_authenticity.h5"
FUZZY_THRESHOLD = 60
TRAIN_DIR = 'data/train'
VAL_DIR = 'data/validation'
BATCH_SIZE = 16
EPOCHS = 7

DOCUMENT_KEYWORDS = {
    'BIR_permit': [
        "Certificate of Registration", "TIN", "Registration Date"
    ],
    'GIS': [
        "General Information Sheet", "Stockholders Information",
        "Board of Directors", "Corporate Secretary", "Principal Office"
    ],
    'financial_statements': [
        "Income Statement","Balance Sheet",  "Assets", "Liabilities", "Equity", "Revenue", 
        "Expenses", "Net Income"
    ]
}

try:
    corrector = pipeline("text2text-generation", model="pszemraj/grammar-synthesis-small")
except Exception as e:
    print(f"Error loading text correction model: {e}")
    corrector = None

def clean_text(text, max_length=255):
    """Clean and correct text using grammar correction"""
    if not text.strip():
        return text
        
    try:
        if corrector is None:
            return text
            
        chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
        corrected = " ".join([
            corrector(f"grammar: {chunk}", max_length=max_length, do_sample=False)[0]['generated_text']
            for chunk in chunks
        ])
        
        patterns = [
            r'REPUBLIKA\s*NG\s*P[I1l]L[I1l]P[I1l]NAS',
            r'REPUBLIKA NG PIIPNAS',
            r'REPUBLIKA NG PIL1PINAS'
            r'C[Ee][Rr][Tt][Ii][Ff][Ii1Ll][Cc][Aa][Tt][Ee]\s*O[Ff]\s*R[Ee][Gg][Ii1l][Ss5][Tt][Rr][Aa][Tt][Ii1l][Oo0][Nn]',
            r'C[Ee][Rr][Tt][Ii][Ff][Ii1Ll][Cc][Aa][Tt][Ee]\s+O[Ff]+\s+R[Ee][Gg][Ii1l][Ss5][Tt][Rr][Aa][Tt][Ii1l][Oo0][Nn]',   
            r'CERT1FICATE\s*OF\s*REG1STRAT10N', 
            r'CERTIF1CATE\s*0F\s*REGISTRATI0N', 
            r'CERIIFICATE\s*OF\s*REGISTRATION', 
            r'CETIFICATE\s*OF\s*REGISTRATION', 
            r'CERTIFICATE\s*OF\s*REGlSTRATION', 
            r'CERTIFICATE\s+0F\s+REGISTRATION',
            r'CERTIF1CATE\s+OF\s+REG1STRAT1ON'  
        ]
        for p in patterns:
            corrected = re.sub(p, 'REPUBLIKA NG PILIPINAS', corrected, flags=re.IGNORECASE)
            
        return corrected
    except Exception as e:
        print(f"Error correcting text: {e}")
        return text

def extract_text_pdf(file_path):
    """Extract text from PDF or image files"""
    text = ""
    if file_path.lower().endswith('.pdf'):
        doc = fitz.open(file_path)
        for page in doc:
            blocks = page.get_text("dict")["blocks"]
            for b in blocks:
                if "lines" in b:
                    for line in b["lines"]:
                        line_text = " ".join([span["text"] for span in line["spans"]])
                        text += line_text.strip() + "\n"
    
    else:
        img = cv2.imread(file_path)
        if img is None:
            raise ValueError(f"Could not read image file: {file_path}")
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        
        text = pytesseract.image_to_string(thresh, config='--psm 6')

    return text

def clean_output(text):
    """Further clean and structure the text"""
    if not text.strip():
        return text
        
    text = re.sub(r'[^a-zA-Z0-9\s\-.,:()%\'\"/]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    corrections = {
        r'REPUBLIKA NG PIIPNAS': 'REPUBLIKA NG PILIPINAS',
        r'PANANALAP!': 'PANANALAPI'
    }
    for pattern, replacement in corrections.items():
        text = re.sub(pattern, replacement, text)

    sections = [
        r'(REPUBLIKA NG PILIPINAS.*?)(?=CERTIFICATE OF REGISTRATION)',
        r'(CERTIFICATE OF REGISTRATION.*?)(?=ACTIVITIES)',
        r'(ACTIVITIES.*?)(?=REMINDERS)',
        r'(REMINDERS.*)'
    ]
    output = [re.search(s, text, re.DOTALL) for s in sections]
    return "\n\n".join([m.group(1).strip() for m in output if m]) or text

def extract_text(image_path, document_type):
    try:

        if document_type == "BIR_permit":
            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
            kernel = np.ones((2,2), np.uint8)
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            result = cv2.bitwise_not(cleaned)

            priortext = pytesseract.image_to_string(result, config='--psm 6 --oem 3 tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-/').strip() 
            sampleText = clean_output (priortext)
            print(sampleText)

            zones = [
                {'label': 'Title', 'coords': (351, 251, 905, 309)}
            ]
            for zone in zones:
                label = zone['label']
                (x1, y1, x2, y2) = zone['coords']
                zone_img = result[y1:y2, x1:x2]

                priortext = pytesseract.image_to_string(zone_img, config='--psm 6 --oem 3 tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-/').strip()
                text = clean_output (priortext)
                print(f"[{label}] {text}")

        elif document_type == "GIS" or document_type == "financial_statements":
            priortext = extract_text_pdf(image_path)
            secondText = clean_text(priortext)
            text = clean_output (secondText)
            print(text)

        return text
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": f"Processing failed: {str(e)}"}


def pdf_to_images(pdf_path):
    """Convert PDF to list of images"""
    doc = fitz.open(pdf_path)
    images = []
    for page in doc:
        pix = page.get_pixmap(dpi=200)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(np.array(img))
    doc.close()
    return images

def load_pretrained_model():
    """Load the pre-trained model"""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train first.")
    return load_model(MODEL_PATH)

import difflib

def find_best_match_in_text(block_text, target_phrase, window_size=40, threshold=0.4):
    best_match = ""
    best_ratio = 0.0
    block_text = block_text.upper()
    target_phrase = target_phrase.upper()

    for i in range(0, len(block_text) - window_size + 1):
        window = block_text[i:i + window_size]
        ratio = difflib.SequenceMatcher(None, window, target_phrase).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = window

    if best_ratio >= threshold:
        return {
            "match_found": True,
            "match_ratio": best_ratio,
            "matched_text": best_match
        }
    else:
        return {
            "match_found": False,
            "match_ratio": best_ratio,
            "matched_text": ""
        }


def verifyDocument(image_path, document_type, model, companyName):
    try:
        if not os.path.exists(image_path):
            return {
                'status': 'error',
                'message': 'File not found'
            }
        
        if image_path.lower().endswith('.pdf'):
            images = pdf_to_images(image_path)
            if not images:
                return {'status': 'error', 'message': 'No images found in the PDF'}
            first_page = images[0]
            image = Image.fromarray(first_page)  
            img = image.resize(IMAGE_SIZE) 
        else:
            img = load_img(image_path, target_size=IMAGE_SIZE)

        x = preprocess_input(np.expand_dims(img_to_array(img), axis=0))

        with open('BACK_END/label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)

        preds = model.predict(x)
        class_idx = np.argmax(preds)
        predicted_class = label_encoder.classes_[class_idx]
        confidence = preds[0][class_idx]

        print("Predicted probabilities:", preds)

        for i, label in enumerate(label_encoder.classes_):
            print(f"{label}: {preds[0][i]*100:.2f}%")

        if document_type == "BIR_permit":
            if any(keyword in predicted_class.lower() for keyword in ['_authentic', '_fake', 'bir']):
                simplified_class = predicted_class.replace('_authentic' or '_fake', '').strip()
            else:
                simplified_class = 'Not a BIR Permit'

        elif document_type == "GIS":
            if any(keyword in predicted_class.lower() for keyword in ['_authentic', '_fake', 'GIS', 'General Information Sheet']):
                simplified_class = predicted_class.replace('_authentic' or '_fake', '').strip()
            else:
                simplified_class = 'Not a General Information Sheet'

        elif document_type == "financial_statements":
            if any(keyword in predicted_class.lower() for keyword in ['_authentic', '_fake', 'Financial Statement', 'FINstatements']):
                simplified_class = predicted_class.replace('_authentic' or '_fake', '').strip()
            else:
                simplified_class = 'Not a Financial Statement'


        if 'Not' in simplified_class:
            final_score = round(float(-confidence) * 100, 2)
        else:
            final_score = round(float(confidence) * 100, 2)


        print(label_encoder.classes_)
        print("Predictions:", preds)
        print("Class index:", class_idx)
        print("Predicted class:", predicted_class)
        
        return {
            'Company Name': companyName,
            'document_type': document_type,
            'predicted_class': simplified_class,
            'confidence': round(float(confidence) * 100, 2),
            'final_score': final_score,
            'status': 'success'
        }

        
    except Exception as e:
        return {
            'status': 'error',  
            'message': str(e)
        }

