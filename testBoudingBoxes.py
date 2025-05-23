import cv2
import matplotlib.pyplot as plt
import pytesseract
import numpy as np
import difflib
import re
import os
import re
import cv2
import numpy as np
import pytesseract
from fuzzywuzzy import fuzz
from transformers import pipeline
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.applications.resnet50 import preprocess_input
import time


image_path = "C:/Users/PLPASIG/Downloads/birTemplateHD (Enhanced).jpg"
img = cv2.imread(image_path)
TIN = '009-028-463-000'
stamp_template = "C:/Users/PLPASIG/Downloads/BIR_stamp.png"
corporate_name = "Center for Local Governance and Professional Devt."

plt.figure(figsize=(12,8))
plt.imshow(img)
plt.axis('off')
plt.show()

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

# Draw rectangles
for zone in zones:
    (x1, y1, x2, y2) = zone['coords']
    cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 0), thickness=2)
    cv2.putText(result, zone['label'], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# Show the image
plt.figure(figsize=(12,8))
plt.imshow(result)
plt.axis('off')
plt.show()

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

for zone in zones:
    label = zone['label']
    (x1, y1, x2, y2) = zone['coords']
    zone_img = result[y1:y2, x1:x2]

    text = pytesseract.image_to_string(zone_img, config='--psm 6').strip()
    print(f"[{label}] {text}")

    if label == 'TIN':

        extracted_tin = ''.join(filter(str.isdigit, text))
        reference_tin = ''.join(filter(str.isdigit, TIN))

        match_ratio = difflib.SequenceMatcher(None, extracted_tin, reference_tin).ratio()
        match_percentage = round(match_ratio * 100, 2)
        print(f"Match Percentage with expected TIN: {match_percentage}%")
    
    if label == 'Taxpayer Name':

        cleaned_text = re.sub(r'\bname\b', '', text, flags=re.IGNORECASE).strip()
        cleaned_reference = re.sub(r'\bname\b', '', corporate_name, flags=re.IGNORECASE).strip()

        extracted_name = cleaned_text.upper()
        reference_name = cleaned_reference.upper()

        match_ratio = difflib.SequenceMatcher(None, extracted_name, reference_name).ratio()
        match_percentage = round(match_ratio * 100, 2)
        print(f"Match Percentage with Corporate Name: {match_percentage}%")

for zone in zone2:
    
    label = zone['label']
    (x1, y1, x2, y2) = zone['coords']
    zone_img = img[y1:y2, x1:x2]

    if label == 'Stamp':

        template = cv2.imread(stamp_template)

        if template.shape > zone_img.shape:
            template = cv2.resize(template, (zone_img.shape[1], zone_img.shape[0]))

        result = cv2.matchTemplate(zone_img, template, cv2.TM_CCOEFF_NORMED)
        _, match_score, _, _ = cv2.minMaxLoc(result)

        match_percentage = round(match_score * 100, 2)
        print(f"Stamp Match Percentage: {match_percentage}%")

    if label == 'Signature':
        has_signature, darkness = is_signature_present(zone_img)
        print(f"Signature Detected: {has_signature} (Dark Pixel %: {darkness}%)")

def load_pretrained_model():
    """Load the pre-trained model"""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train first.")
    return load_model(MODEL_PATH)

IMAGE_SIZE = (224, 224)
MODEL_PATH = "resnet50_authenticity.h5"

model = load_pretrained_model()  

img = load_img(image_path, target_size=IMAGE_SIZE)
x = preprocess_input(np.expand_dims(img_to_array(img), axis=0))

preds = model.predict(x)
class_labels = ['authentic', 'fake']  
class_idx = np.argmax(preds)
print("Model summary:")
model.summary()
print("Prediction output:", preds)
print("Prediction shape:", preds.shape)

predicted_class = class_labels[class_idx]
confidence = preds[0][class_idx]

print()
print(confidence)
print(f"Match Percentage with expected TIN: {match_percentage}%")
print(f"Match Percentage with Corporate Name: {match_percentage}%")
print(f"Stamp Match Percentage: {match_percentage}%")
print(f"Signature Detected: {has_signature} (Dark Pixel %: {darkness}%)")
