from datetime import datetime
import re
import fitz
from fuzzywuzzy import fuzz
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
import os
import pytesseract
import cv2

# Constants
IMAGE_SIZE = (224, 224)
MODEL_PATH = "resnet50_authenticity.h5"
FUZZY_THRESHOLD = 80
TRAIN_DIR = 'data/train'
VAL_DIR = 'data/validation'
BATCH_SIZE = 16
EPOCHS = 7

DOCUMENT_KEYWORDS = {
    'financial_statements': [
        "Balance Sheet", "Income Statement",
        "Assets", "Liabilities", "Equity", "Revenue", "Expenses", "Net Income"
    ]
}

def assess_structural_similarity(text, document_type='financial_statements'):
    """Enhanced structural similarity assessment with value verification"""
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


image_path ="C:/Users/PLPASIG/Downloads/sample_financial_statements (with date).pdf"
global_text = ""



IMAGE_SIZE = (224, 224)
MODEL_PATH = "resnet50_authenticity.h5"

def load_pretrained_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train it first.")
    return load_model(MODEL_PATH)

def predict_authenticity(image_path, document_type, model, companyName):
    """Use ResNet50 model to predict document authenticity."""
    img = load_img(image_path, target_size=IMAGE_SIZE)
    x = preprocess_input(np.expand_dims(img_to_array(img), axis=0))

    preds = model.predict(x)
    class_labels = ['authentic', 'fake']
    class_idx = np.argmax(preds)
    predicted_class = class_labels[class_idx]
    confidence = float(preds[0][class_idx])

    ocr_text = extract_text(image_path)

    expected_keywords = DOCUMENT_KEYWORDS.get(document_type, [])

    missing_keywords = []
    for kw in expected_keywords:
        score = fuzz.partial_ratio(kw.lower(), ocr_text.lower())
        if score < FUZZY_THRESHOLD:
            missing_keywords.append(kw)

    text_score = 1 - (len(missing_keywords)/len(expected_keywords)) if expected_keywords else 1
    final_score = f"{round((confidence * 0.6 + text_score * 0.4) * 100, 2)}%"
    financial_assessment(image_path)
    return {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'status': 'success'
    }

    
def financial_assessment (image_path):
    global global_text
    try:
        text = extract_text(image_path)

        document_date = extract_year(text)

        structure_score = assess_structural_similarity(text)

        global_text += text
        current_assets = extract_amount("Total Current Assets")
        current_liabilities = extract_amount("Current Liabilities")
        inventory = extract_amount("Inventory")
        total_liabilities = extract_amount("Total Liabilities")
        total_assets = extract_amount("Total Assets")
        total_equity = (total_assets - total_liabilities)
        revenue = extract_amount("Revenue")
        net_income = extract_amount("Net Income")
        current_ratio = (current_assets/current_liabilities)
        quick_ratio = ((current_assets -inventory) / current_liabilities)
        debt_to_equity_ratio = (total_liabilities/total_equity)
        netProfit_margin = (net_income/revenue)

        print("This are the results.")
        print()
        print(f"Current Ratio: {current_ratio:.2f}")
        print(f"Quick Ratio: {quick_ratio:.2f}")
        print(f"Debt to Equity Ratio: {debt_to_equity_ratio:.2f}")
        print(f"Net Profit Margin: {netProfit_margin:.2%}")


        return
    
    except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }

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

def extract_text(file_path):
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


financial_assessment(image_path)
