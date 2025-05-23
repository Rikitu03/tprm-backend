import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.applications.resnet50 import preprocess_input
import tensorflow as tf
import pytesseract
import cv2
import re
import spacy
from fuzzywuzzy import fuzz
from transformers import pipeline
from keybert import KeyBERT
import re


def extract_text(image_path):
    img = cv2.imread(image_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    scaled = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
    base, ext = os.path.splitext(image_path)
    processed_path = f"{base}_processed{ext}"
    
    custom_config = r'--oem 3 --psm 6'
    extracted_text = pytesseract.image_to_string(scaled, config=custom_config, lang='eng')

    print("→ Extracted OCR Text:")

    return extracted_text

corrector = pipeline("text2text-generation", model="pszemraj/grammar-synthesis-small")

def clean_text(text, max_length=None):
    if not text.strip():
        return text
        
    try:
        chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
        corrected_chunks = []

        for chunk in chunks:
            corrected = corrector(f"grammar: {chunk}", max_length=max_length, do_sample=False)[0]['generated_text']
            corrected_chunks.append(corrected.replace("grammar: ", ""))
        
        full_corrected = " ".join(corrected_chunks)

        raw_text = full_corrected

        republika_variations = [
            r'REPUBLIKA NG PIIPNAS',
            r'REPUBLIKA NG PIL1PINAS',
            r'REPUBLIKA NG P[I1l]L[I1l]P[I1l]NAS',
            r'REPUBL[I1l]KA NG PILIPINAS',
            r'REPUBLIKA\s*NG\s*P[!|]LIPINAS',
        ]

        for variation in republika_variations:
            raw_text = re.sub(
                variation, 
                'REPUBLIKA NG PILIPINAS', 
                raw_text, 
                flags=re.IGNORECASE
            )

        match = re.search(r'REPUBLIKA NG PILIPINAS', raw_text, re.IGNORECASE)
        return match.group(0) if match else "REPUBLIKA NG PILIPINAS not found"

    except Exception as e:
        print(f"Error correcting text: {str(e)}")
        return text




def clean_output(text):
    """Process OCR output to make it more readable without intros/outros"""
    
    text = re.sub(r'[^a-zA-Z0-9\s\-.,:()%\'\"/]', '', text) 
    text = re.sub(r'\s+', ' ', text).strip()  
    
    corrections = {
        r'REPUBLIKA NG PIIPNAS': 'REPUBLIKA NG PILIPINAS',
        r'PANANALAP!': 'PANANALAPI',
        r'RENTAS INTERNAS': 'RENTAS INTERNAS',
        r'Decerder (\d+)': r'December \1',
        r'(\d+)RC(\d+)': r'\1RC\2'  
    }
    
    for pattern, replacement in corrections.items():
        text = re.sub(pattern, replacement, text)
    
    sections = [
        r'(REPUBLIKA NG PILIPINAS.*?)(?=CERTIFICATE OF REGISTRATION)',
        r'(CERTIFICATE OF REGISTRATION.*?)(?=ACTIVITIES)',
        r'(ACTIVITIES.*?)(?=REMINDERS)',
        r'(REMINDERS.*)'
    ]
    
    cleaned = []
    for section in sections:
        match = re.search(section, text, re.DOTALL)
        if match:
            cleaned.append(match.group(1).strip())
    
    return "\n\n".join(cleaned) if cleaned else text


image_path = "C:/Users/PLPASIG/Downloads/testBIR.jpg"
print(f"\n📄 OCR TEXT EXTRACTION:")
ocr_text = extract_text(image_path)
cleaned_ocr_text = clean_text(ocr_text)
final_output = clean_output(cleaned_ocr_text)
print(f"{final_output}")

