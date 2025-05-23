import os
import re
import cv2
import numpy as np
import pytesseract
from fuzzywuzzy import fuzz
from transformers import pipeline
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
import mysql.connector
from datetime import datetime
import fitz
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import re
import cv2
import numpy as np
import pytesseract
from fuzzywuzzy import fuzz
from transformers import pipeline
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.applications.resnet50 import preprocess_input
import pickle
from urllib.parse import urlparse

db_url = urlparse(os.getenv('MYSQL_URL', 'mysql://root:uOgCDJxppVVIHkEYJWBUHJHlAHyrmuMr@nozomi.proxy.rlwy.net:30007/railway'))

DB_CONFIG = {
    'host': db_url.hostname or 'localhost',
    'user': db_url.username or 'root',
    'password': db_url.password or '',
    'database': db_url.path.lstrip('/') or 'tprm',
    'port': db_url.port or 3306,
    'charset': 'utf8mb4'
}

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""Financial Statement PROCESSES"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def predict_document(image_path, document_type, model, companyName):
    """Predict document authenticity and validate content"""
    try:
        if document_type == 'BIR_permit':
            return process_bir_permit(image_path, document_type, model, companyName)
        elif document_type == 'GIS':
            return process_gis(image_path, document_type, model, companyName)
        elif document_type == 'financial_statements':
            return process_financialStatements(image_path, document_type, model, companyName)


    except Exception as e:
        return {
            'status': 'error',
            'message': str(e)
        }   

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""UTILITIES WITH DATABASE"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def get_vendor_info(company_name):
    """Fetch vendor information like TIN and Company Name."""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)

        query = "SELECT TIN_number, company_name, SEC_number FROM vendors WHERE company_name = %s"
        cursor.execute(query, (company_name,))
        vendor = cursor.fetchone()

        cursor.close()
        conn.close()

        return vendor  
    except mysql.connector.Error as e:
        print(f"Database error: {e}")
        return None

def get_vendor_GIS_info(company_name):
    """Fetch vendor information like TIN and Company Name."""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)

        query = "SELECT TIN_number, SEC_number, company_name FROM vendors WHERE company_name = %s"
        cursor.execute(query, (company_name,))
        vendor = cursor.fetchone()

        cursor.close()
        conn.close()

        return vendor  
    except mysql.connector.Error as e:
        print(f"Database error: {e}")
        return None


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""OTHER UTILITIES"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def is_recent_date(date_str):
    try:
        parsed_date = datetime.strptime(date_str, "%B %d, %Y")
    except ValueError:
        try:
            parsed_date = datetime.strptime(date_str, "%m/%d/%Y")
        except ValueError:
            return False  
    return (datetime.now() - parsed_date).days <= 730  

def is_signature_present(zone_img, threshold=0.02):
    """
    Determines if signature is present based on dark pixel ratio.
    - `zone_img` must be a grayscale image.
    - `threshold` is the minimum dark pixel ratio to consider a signature present.
    """
    gray = cv2.cvtColor(zone_img, cv2.COLOR_BGR2GRAY) if len(zone_img.shape) == 3 else zone_img
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

    dark_pixels = cv2.countNonZero(binary)
    total_pixels = binary.shape[0] * binary.shape[1]
    dark_ratio = dark_pixels / total_pixels

    return dark_ratio > threshold, round(dark_ratio * 100, 2)


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""CONSTANTS"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

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

# Constants
IMAGE_SIZE = (512, 512)
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
MODEL_PATH = os.path.join(BASE_DIR, "resnet50_authenticity.h5")
FUZZY_THRESHOLD = 80
TRAIN_DIR = 'data/train'
VAL_DIR = 'data/validation'
BATCH_SIZE = 16
EPOCHS = 7

try:
    corrector = pipeline("text2text-generation", model="pszemraj/grammar-synthesis-small")
except Exception as e:
    print(f"Error loading text correction model: {e}")
    corrector = None
    

def extract_text(image_path, companyName, document_type):
    import difflib

    tin_match_percentage = 0
    corporateName_match_percentage = 0
    stamp_match_percentage = 0
    has_signature = False

    """Extract text from image using OCR"""
    try:
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((2,2), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        result = cv2.bitwise_not(cleaned)

        zones = [
            {'label': 'Header', 'coords': (421, 30, 852, 178)},
            {'label': 'Title', 'coords': (351, 251, 905, 309)},
            {'label': 'Taxpayer Name', 'coords': (353, 337, 903, 418)},
            {'label': 'TIN', 'coords': (22, 336, 352, 418)},
            {'label': 'Line of Business', 'coords': (563, 721, 1119, 875)},
            {'label': 'Date Issued', 'coords': (981, 920, 1219, 1005)},
            {'label': 'Registration Date', 'coords': (904, 336, 1244, 419)}
        ]

        zone2 = [
            {'label': 'Stamp', 'coords': (29, 1347, 304, 1493)},
            {'label': 'Signature', 'coords': (545, 1136, 1262, 1599)}
        ]

        vendor_info = get_vendor_info(companyName)

        for zone in zones:
            label = zone['label']
            (x1, y1, x2, y2) = zone['coords']
            zone_img = result[y1:y2, x1:x2]

            text = pytesseract.image_to_string(zone_img, config='--psm 6').strip()
            print(f"[{label}] {text}")

            if label == 'TIN':

                extracted_tin = ''.join(filter(str.isdigit, text))
                reference_tin = ''.join(filter(str.isdigit, vendor_info['TIN_number']))

                match_ratio = difflib.SequenceMatcher(None, extracted_tin, reference_tin).ratio()
                tin_match_percentage = round(match_ratio * 100, 2)
                print(f"Match Percentage with expected TIN: {tin_match_percentage}%")
            
            if label == 'Taxpayer Name':

                cleaned_text = re.sub(r'\bname\b', '', text, flags=re.IGNORECASE).strip()
                cleaned_reference = re.sub(r'\bname\b', '', vendor_info['company_name'], flags=re.IGNORECASE).strip()

                extracted_name = cleaned_text.upper()
                reference_name = cleaned_reference.upper()

                match_ratio = difflib.SequenceMatcher(None, extracted_name, reference_name).ratio()
                corporateName_match_percentage = round(match_ratio * 100, 2)
                print(f"Match Percentage with Corporate Name: {corporateName_match_percentage}%")

        for zone in zone2:
            label = zone['label']
            (x1, y1, x2, y2) = zone['coords']
            zone_img = img[y1:y2, x1:x2]

            if label == 'Stamp':
                stamp_template = "C:/Users/PLPASIG/Downloads/BIR_stamp.png"
                template = cv2.imread(stamp_template)

                if template.shape > zone_img.shape:
                    template = cv2.resize(template, (zone_img.shape[1], zone_img.shape[0]))

                result = cv2.matchTemplate(zone_img, template, cv2.TM_CCOEFF_NORMED)
                _, match_score, _, _ = cv2.minMaxLoc(result)

                stamp_match_percentage = round(match_score * 100, 2)
                print(f"Stamp Match Percentage: {stamp_match_percentage}%")

            if label == 'Signature':
                has_signature, darkness = is_signature_present(zone_img)
                print(f"Signature Detected: {has_signature} (Dark Pixel %: {darkness}%)")

        return {
            'TIN_match': tin_match_percentage,
            'CorporateName_match': corporateName_match_percentage,
            'Stamp_match': stamp_match_percentage,
            'Signature_present': has_signature
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            'TIN_match': 0,
            'CorporateName_match': 0,
            'Stamp_match': 0,
            'Signature_present': False
        }


def clean_text(text, max_length=256):
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
        ]
        for p in patterns:
            corrected = re.sub(p, 'REPUBLIKA NG PILIPINAS', corrected, flags=re.IGNORECASE)
            
        return corrected
    except Exception as e:
        print(f"Error correcting text: {e}")
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

def load_pretrained_model():
    """Load the pre-trained model"""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train first.")
    return load_model(MODEL_PATH)


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""Financial Statement PROCESSES"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

global_text = ""

def assess_structural_similarity(text, document_type='financial_statements'):
    """Enhanced structural similarity assessment with value verification"""

    if not text.strip():
        return 0.0
    
    extracted_values = {}
    
    def get_cached_amount(label):
        if label not in extracted_values:
            extracted_values[label] = extract_amount(label, verbose=False)
        return extracted_values[label]
    
    keyword_checks = {

        "Balance Sheet": (3, lambda: fuzz.partial_ratio("Balance Sheet", text) >= FUZZY_THRESHOLD),
        "Income Statement": (3, lambda: fuzz.partial_ratio("Income Statement", text) >= FUZZY_THRESHOLD),
        
        "Assets": (2, lambda: any(fuzz.partial_ratio(x, text) >= FUZZY_THRESHOLD 
                                 for x in ["Assets", "ASSETS", "Asset"])),
        "Liabilities": (2, lambda: any(fuzz.partial_ratio(x, text) >= FUZZY_THRESHOLD 
                                     for x in ["Liabilities", "LIABILITIES", "Liability"])),
        "Equity": (2, lambda: any(fuzz.partial_ratio(x, text) >= FUZZY_THRESHOLD 
                               for x in ["Equity", "EQUITY", "Shareholder's Equity"])),
        
        "Total Current Assets": (1, lambda: get_cached_amount("Total Current Assets") is not None),
        "Current Liabilities": (1, lambda: get_cached_amount("Current Liabilities") is not None),
        "Revenue": (1, lambda: get_cached_amount("Revenue") is not None),
        "Net Income": (1, lambda: get_cached_amount("Net Income") is not None),
        "Total Assets": (1, lambda: get_cached_amount("Total Assets") is not None),
        "Inventory": (1, lambda: get_cached_amount("Inventory") is not None),
        "Total Liabilities": (1, lambda: get_cached_amount("Total Liabilities") is not None),
        "Expenses": (1, lambda: any(fuzz.partial_ratio(x, text) >= FUZZY_THRESHOLD 
                                   for x in ["Expenses", "EXPENSES", "Operating Expenses"]))
    }
    
    total_weight = sum(w for w, _ in keyword_checks.values())
    matched_weight = 0
    
    print("\n=== Structural Similarity Assessment ===")
    for keyword, (weight, verify) in keyword_checks.items():
        if verify():
            matched_weight += weight
            print(f"✓ Found '{keyword}' (weight: {weight})")
        else:
            print(f"✗ Missing '{keyword}' (weight: {weight})")
    
    similarity_score = (matched_weight / total_weight) * 100
    return round(similarity_score, 2)

def extract_amount(label, verbose=True):
    """Modified extract_amount with optional verbosity"""
    if verbose:
        print(f"Looking for: {label}")
    
    lines = global_text.splitlines() 
    for i, line in enumerate(lines):
        if label.lower() in line.lower(): 
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                match = re.search(r"PHP\s*([\d,]+)", next_line)
                if match:
                    amount = int(match.group(1).replace(",", ""))
                    if verbose:
                        print(f"Found {label}: {amount}")
                    return amount
    if verbose:
        print(f"Could not find {label} in the text.")
    return None


def merge_lines(text):
    lines = text.split(" ")
    merged_lines = []
    
    for line in lines:
        if line.strip().startswith("PHP"):
            merged_lines[-1] += f" {line.strip()}"
        else:
            merged_lines.append(line.strip())
    
    return " ".join(merged_lines)

def extract_text_financial(file_path):
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

def extract_year(text):
    """Extract year from document text"""

    date_match = re.search(r"Date:\s*([A-Za-z]+\s\d{1,2},\s\d{4})", text)
    if date_match:
        full_date = date_match.group(1)
        try:
            return datetime.strptime(full_date, "%B %d, %Y").year 
        except ValueError:
            pass
    
    year_match = re.search(r"\b(20\d{2})\b", text)  
    return int(year_match.group(1)) if year_match else None


def interpret_kri_financial(score):
    if score <= 50:
        return {'Interpretation': "Very Bad", 'Severity': "Critical"}
    elif 51 <= score <= 75:
        return {'Interpretation': "Bad", 'Severity': "High"}
    elif 76 <= score <= 85:
        return {'Interpretation': "Adequate", 'Severity': "Moderate"}
    elif 86 <= score <= 94:
        return {'Interpretation': "Good", 'Severity': "Low"}
    else:
        return {'Interpretation': "Very Good", 'Severity': "Minimal"}

def interpret_date_financial(date):
    if not date:  
        return {'Interpretation': "No date found", 'Severity': "Critical", 'Score': 0}
    if date == "Current year": 
        return {'Interpretation': "Very Good", 'Severity': "Minimal", 'Score': 100}
    elif date == "1 year ago":
        return {'Interpretation': "Good", 'Severity': "Low", 'Score': 90}
    elif date == "2 years ago":
        return {'Interpretation': "Adequate", 'Severity': "Moderate", 'Score': 70}
    elif date == "3 years ago":
        return {'Interpretation': "Bad", 'Severity': "High", 'Score': 50}
    elif date == "4 years ago":
        return {'Interpretation': "Very", 'Severity': "Critical", 'Score': 30}
    elif date == "Older than 4 years (or future date)":
        return {'Interpretation': "Unacceptable", 'Severity': "Unacceptable", 'Score': 0}
    
def process_financialStatements(image_path, document_type, model, companyName):
    try:    
        images = pdf_to_images(image_path)
        if not images:
            return {'status': 'error', 'message': 'No images found in the PDF'}

        first_page_img = images[0]

        with open('BACK_END/label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)

        img = Image.fromarray(first_page_img)  
        img = img.resize(IMAGE_SIZE) 
        x = preprocess_input(np.expand_dims(img_to_array(img), axis=0))

        preds = model.predict(x)
        class_idx = np.argmax(preds)
        predicted_class = label_encoder.classes_[class_idx]
        confidence = preds[0][class_idx]

        if 'authentic' in predicted_class.lower():
            simplified_class = 'authentic'
        else:
            simplified_class = 'fake'

        global global_text
        text = extract_text_financial(image_path)
        global_text = text

        document_date = extract_year(text)

        if document_date:
            extracted_year = int(document_date)
            current_year = datetime.now().year
            year_diff = current_year - extracted_year

            if year_diff == 0:
                date_match_percentage = "Current year"
            elif year_diff == 1:
                date_match_percentage = "1 year ago"
            elif year_diff == 2:
                date_match_percentage = "2 years ago"
            elif year_diff == 3:
                date_match_percentage = "3 years ago"
            elif year_diff == 4:
                date_match_percentage = "4 years ago"
            else:
                date_match_percentage = "Older than 4 years (or future date)"
        
        else:
            print("Could not extract year from date.")
        
        date_result = interpret_date_financial(date_match_percentage)
        dateScore = date_result['Score']
        structure_score = assess_structural_similarity(text)
        structure_result = interpret_kri_financial(structure_score)

        if simplified_class == 'authentic':
            confidence_score = confidence * 100
        elif simplified_class == 'fake':
            confidence_score = -(confidence * 100)

        final_score = round(confidence_score * 0.4 + dateScore * 0.3 + structure_score * 0.3, 2)

        return {
            'document_type': document_type,
            'predicted_class': simplified_class,
            'confidence': round(float(confidence) * 100, 2),
            'status': 'success',
            'final_score': final_score,
            'risk_indicators': [
                {
                    'name': 'Structural Similarity',
                    'value': structure_score,
                    'severity': structure_result['Severity'],
                    'interpretation': structure_result['Interpretation']
                },
                {
                    'name': 'Date Recency',
                    'value': date_match_percentage,
                    'severity': date_result['Severity'],
                    'interpretation': date_result['Interpretation']
                }
            ]
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e)
        }



"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""BIR PROCESSES"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def interpret_kri_bir(score):
    if score <= 50:
        return {'Interpretation': "Very Bad", 'Severity': "Critical"}
    elif 51 <= score <= 75:
        return {'Interpretation': "Bad", 'Severity': "High"}
    elif 76 <= score <= 85:
        return {'Interpretation': "Adequate", 'Severity': "Moderate"}
    elif 86 <= score <= 94:
        return {'Interpretation': "Good", 'Severity': "Low"}
    else:
        return {'Interpretation': "Very Good", 'Severity': "Minimal"}

def interpret_signature_bir(status):
    if status:  
        return {'Interpretation': "Very Good", 'Severity': "Minimal"}
    else:
        return {'Interpretation': "Bad", 'Severity': "High"}

def process_bir_permit(image_path, document_type, model, companyName):
    """Predict document authenticity and validate content"""
    try:
        if not os.path.exists(image_path):
            return {
                'status': 'error',
                'message': 'File not found'
            }
        
        img = load_img(image_path, target_size=IMAGE_SIZE)
        x = preprocess_input(np.expand_dims(img_to_array(img), axis=0))

        with open('BACK_END/label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)

        preds = model.predict(x)
        class_idx = np.argmax(preds)
        predicted_class = label_encoder.classes_[class_idx]
        confidence = preds[0][class_idx]

        if 'authentic' in predicted_class.lower():
            simplified_class = 'authentic'
        else:
            simplified_class = 'fake'

        ocr_text = extract_text(image_path, companyName, document_type)
        print(ocr_text)

        nameScore = ocr_text['CorporateName_match']
        tinScore = ocr_text['TIN_match']
        stampScore = ocr_text['Stamp_match']
        signatureStatus = ocr_text['Signature_present']

        signatureScore = 100 if ocr_text['Signature_present'] == 'True' else 0

        if simplified_class == 'authentic':
            confidence_score = confidence * 100
        elif simplified_class == 'fake':
            confidence_score = -(confidence * 100)

        final_score = round(confidence_score * 0.2 + nameScore * 0.2 + tinScore * 0.2 + stampScore * 0.2 + signatureScore * 0.2, 2)

        Corporate_matching_interpretation = interpret_kri_bir(nameScore)
        TIN_Matching_interpretation = interpret_kri_bir(tinScore)
        Stamp_Matching_interpretation = interpret_kri_bir(stampScore)
        Signature_interpretation  = interpret_signature_bir(signatureStatus)

        return {
            'document_type': document_type,
            'predicted_class': simplified_class,
            'confidence': round(float(confidence) * 100, 2),
            'status': 'success',
            'final_score': final_score,
            'risk_indicators': [
                {
                    'name': 'Corporate Name Match',
                    'value': nameScore,
                    'severity': Corporate_matching_interpretation['Severity'],
                    'interpretation': Corporate_matching_interpretation['Interpretation']
                },
                {
                    'name': 'TIN Match',
                    'value': tinScore,
                    'severity': TIN_Matching_interpretation['Severity'],
                    'interpretation': TIN_Matching_interpretation['Interpretation']
                },
                {
                    'name': 'Stamp Match',
                    'value': stampScore,
                    'severity': Stamp_Matching_interpretation['Severity'],
                    'interpretation': Stamp_Matching_interpretation['Interpretation']
                },
                {
                    'name': 'Signature Presence',
                    'value': 100 if signatureStatus else 0,
                    'severity': Signature_interpretation['Severity'],
                    'interpretation': Signature_interpretation['Interpretation']
                }
            ]
        }

    except Exception as e:
        return {
            'status': 'error',
            'message': f'Processing failed: {str(e)}'
        }

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""GIS PROCESSES"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""  
def interpret_kri_gis(score):
    if score <= 50:
        return {'Interpretation': "Very Bad", 'Severity': "Critical"}
    elif 51 <= score <= 75:
        return {'Interpretation': "Bad", 'Severity': "High"}
    elif 76 <= score <= 85:
        return {'Interpretation': "Adequate", 'Severity': "Moderate"}
    elif 86 <= score <= 94:
        return {'Interpretation': "Good", 'Severity': "Low"}
    else:
        return {'Interpretation': "Very Good", 'Severity': "Minimal"}

def interpret_date_gis(date):
    if not date:  
        return {'Interpretation': "No date found", 'Severity': "Critical", 'Score': 0}
    if date == "Current year":  
        return {'Interpretation': "Very Good", 'Severity': "Minimal", 'Score': 100}
    elif date == "1 year ago":
        return {'Interpretation': "Good", 'Severity': "Low", 'Score': 90}
    elif date == "2 years ago":
        return {'Interpretation': "Adequate", 'Severity': "Moderate", 'Score': 70}
    elif date == "3 years ago":
        return {'Interpretation': "Bad", 'Severity': "High", 'Score': 50}
    elif date == "4 years ago":
        return {'Interpretation': "Very Bad", 'Severity': "Critical", 'Score': 30}
    elif date == "Older than 4 years (or future date)":
        return {'Interpretation': "Unacceptable", 'Severity': "Unacceptable", 'Score': 0}

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

def extract_with_bboxes(image, regions, companyName, display=True):
    """Extract text from specified regions and optionally display with bounding boxes"""
    import difflib
    from datetime import datetime
    
    extracted_data = {
        'CorporateName_match': 0,
        'TIN_match': 0,
        'sec_match': 0,
        'date_match': "No date extracted"
    }

    #corporateName_match_percentage = 0
    #tin_match_percentage = 0
    #sec_match_percentage = 0
    #date_match_percentage = "No date extracted"

    vendor_info = get_vendor_info(companyName)
    if not vendor_info:
        return extracted_data
    
    vendor_info = get_vendor_info(companyName)
    
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    
    if display:
        fig, ax = plt.subplots(figsize=(15, 20))
        ax.imshow(image)
    
    extracted_data = {}
    
    for region_name, (x, y, w, h) in regions.items():
        if display:
            rect = patches.Rectangle(
                (x, y), w, h, 
                linewidth=2, 
                edgecolor='r', 
                facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(x, y-10, region_name, color='red', fontsize=12)
        
        roi = image[y:y+h, x:x+w]
        text = pytesseract.image_to_string(roi, config='--psm 6').strip()
        extracted_data[region_name] = text

        if region_name == "Corporate Name":
            cleaned_text = re.sub(r'\bname\b', '', text, flags=re.IGNORECASE).strip()
            cleaned_reference = re.sub(r'\bname\b', '', companyName, flags=re.IGNORECASE).strip()

            extracted_name = cleaned_text.upper()
            reference_name = cleaned_reference.upper()

            match_ratio = difflib.SequenceMatcher(None, extracted_name, reference_name).ratio()
            corporateName_match_percentage = round(match_ratio * 100, 2)
            print(f"Match Percentage with Corporate Name: {corporateName_match_percentage}%")
        
        elif region_name == "TIN":
            extracted_tin = ''.join(filter(str.isdigit, text))
            reference_tin = ''.join(filter(str.isdigit, vendor_info['TIN_number']))

            match_ratio = difflib.SequenceMatcher(None, extracted_tin, reference_tin).ratio()
            tin_match_percentage = round(match_ratio * 100, 2)
            print(f"Match Percentage with expected TIN: {tin_match_percentage}%")
        
        elif region_name == "Date":
            year_match = re.search(r'\b(20\d{2})\b', text) 
            if year_match:
                extracted_year = int(year_match.group(1))
                current_year = datetime.now().year
                year_diff = current_year - extracted_year

                if year_diff == 0:
                    date_match_percentage = "Current year"
                elif year_diff == 1:
                    date_match_percentage = "1 year ago"
                elif year_diff == 2:
                    date_match_percentage = "2 years ago"
                elif year_diff == 3:
                    date_match_percentage = "3 years ago"
                elif year_diff == 4:
                    date_match_percentage = "4 years ago"
                else:
                    date_match_percentage = "Older than 4 years (or future date)"
                
                print(f"Date Match Percentage: {date_match_percentage}, Year: {extracted_year}")
            else:
                print("Could not extract year from date.")

        elif region_name == "SEC Registration":
            cleaned_text = re.sub(r'\bname\b', '', text, flags=re.IGNORECASE).strip()
            cleaned_reference = re.sub(r'\bname\b', '', vendor_info['SEC_number'], flags=re.IGNORECASE).strip()

            extracted_name = cleaned_text.upper()
            reference_name = cleaned_reference.upper()

            match_ratio = difflib.SequenceMatcher(None, extracted_name, reference_name).ratio()
            sec_match_percentage = round(match_ratio * 100, 2)
            print(f"Match Percentage with SEC: {sec_match_percentage}%")
    
    return {
        'CorporateName_match': corporateName_match_percentage,
        'TIN_match': tin_match_percentage,
        'sec_match': sec_match_percentage,
        'date_match': date_match_percentage
    }
    
def process_gis(image_path, document_type, model, companyName):
    """Predict document authenticity and validate content"""
    try:

        images = pdf_to_images(image_path)
        if not images:
            return {'status': 'error', 'message': 'No images found in the PDF'}

        first_page_img = images[0]

        with open('BACK_END/label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)

        img = Image.fromarray(first_page_img)  
        img = img.resize(IMAGE_SIZE) 
        x = preprocess_input(np.expand_dims(img_to_array(img), axis=0))

        preds = model.predict(x)
        class_idx = np.argmax(preds)
        predicted_class = label_encoder.classes_[class_idx]
        confidence = preds[0][class_idx]

        if 'authentic' in predicted_class.lower():
            simplified_class = 'authentic'
        else:
            simplified_class = 'fake'

        regions = {
            "Corporate Name": (411, 814, 682, 71),  
            "SEC Registration": (411, 997, 674, 98),
            "TIN": (1101, 1051, 502, 44),
            "Date": (840,114,218,36)
        }

        extracted_data = extract_with_bboxes(first_page_img, regions, companyName)

        print("\nExtracted Information:")
        for key, value in extracted_data.items():
            print(f"{key}: {value}")        
        
        nameScore = extracted_data['CorporateName_match']
        tinScore = extracted_data['TIN_match']
        secScore = extracted_data['sec_match']
        date_result = interpret_date_gis(extracted_data['date_match'])
        dateScore = date_result['Score']

        if simplified_class == 'authentic':
            confidence_score = confidence * 100
        elif simplified_class == 'fake':
            confidence_score = -(confidence * 100)

        final_score = round(confidence_score * 0.2 + nameScore * 0.2 + tinScore * 0.2 + secScore * 0.2 + dateScore * 0.2, 2)

        Corporate_matching_interpretation = interpret_kri_gis(nameScore)
        TIN_Matching_interpretation = interpret_kri_gis(tinScore)
        SEC_Matching_interpretation = interpret_kri_gis(secScore)
        date_interpretation  = interpret_date_gis(extracted_data['date_match'])

        return {
            'document_type': document_type,
            'predicted_class': simplified_class,
            'confidence': round(float(confidence) * 100, 2),
            'status': 'success',
            'final_score': final_score,
            'risk_indicators': [
                {
                    'name': 'Corporate Name Match',
                    'value': nameScore,
                    'severity': Corporate_matching_interpretation['Severity'],
                    'interpretation': Corporate_matching_interpretation['Interpretation']
                },
                {
                    'name': 'TIN Match',
                    'value': tinScore,
                    'severity': TIN_Matching_interpretation['Severity'],
                    'interpretation': TIN_Matching_interpretation['Interpretation']
                },
                {
                    'name': 'SEC Match',
                    'value': secScore,
                    'severity': SEC_Matching_interpretation['Severity'],
                    'interpretation': SEC_Matching_interpretation['Interpretation']
                },
                {
                    'name': 'Date Recency',
                    'value': extracted_data['date_match'],
                    'severity': date_interpretation['Severity'],
                    'interpretation': date_interpretation['Interpretation']
                }
            ]
        }

        
    except Exception as e:
        return {
            'status': 'error',  
            'message': str(e)
        }
