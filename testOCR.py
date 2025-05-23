import cv2
import os
import math
from collections import Counter
from google.cloud import vision
import re

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "C:/Users/PLPASIG/Downloads/still-function-457007-p4-2e3120dac3f1.json"

def extract_text(image_path):

    img = cv2.imread(image_path)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    _, encoded_image = cv2.imencode('.jpg', rgb)
    content = encoded_image.tobytes()

    client = vision.ImageAnnotatorClient()

    image = vision.Image(content=content)

    response = client.text_detection(image=image)

    if response.text_annotations:
        extracted_text = response.text_annotations[0].description
    else:
        extracted_text = ""

    return extracted_text

extracted_text = extract_text("C:/Users/PLPASIG/Downloads/testBIR.jpg")
print("→ Extracted OCR Text:")
print(extracted_text)

