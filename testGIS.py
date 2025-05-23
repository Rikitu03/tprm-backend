from datetime import datetime
import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image
import pytesseract
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import difflib
import re

corporate_name = "Starbucks Incorporation"
TIN = "000-123-324-"
SEC = "CN2021000123"

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

def extract_with_bboxes(image, regions, display=True):
    """Extract text from specified regions and optionally display with bounding boxes"""
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    
    fig, ax = plt.subplots(figsize=(15, 20))
    ax.imshow(image)
    
    extracted_data = {}
    
    for region_name, (x, y, w, h) in regions.items():
        rect = patches.Rectangle(
            (x, y), w, h, 
            linewidth=2, 
            edgecolor='r', 
            facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(x, y-10, region_name, color='red', fontsize=12)
        
        roi = image[y:y+h, x:x+w]
        text = pytesseract.image_to_string(roi, config='--psm 6')
        extracted_data[region_name] = text.strip()

        if region_name == "Corporate Name":
            cleaned_text = re.sub(r'\bname\b', '', text, flags=re.IGNORECASE).strip()
            cleaned_reference = re.sub(r'\bname\b', '', corporate_name, flags=re.IGNORECASE).strip()

            extracted_name = cleaned_text.upper()
            reference_name = cleaned_reference.upper()

            match_ratio = difflib.SequenceMatcher(None, extracted_name, reference_name).ratio()
            corporateName_match_percentage = round(match_ratio * 100, 2)
            print(f"Match Percentage with Corporate Name: {corporateName_match_percentage}%")
        
        if region_name == "TIN":
            extracted_tin = ''.join(filter(str.isdigit, text))
            reference_tin = ''.join(filter(str.isdigit, TIN))

            match_ratio = difflib.SequenceMatcher(None, extracted_tin, reference_tin).ratio()
            tin_match_percentage = round(match_ratio * 100, 2)
            print(f"Match Percentage with expected TIN: {tin_match_percentage}%")
        
        if region_name == "Date":
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

        if region_name == "SEC Registration":
            cleaned_text = re.sub(r'\bname\b', '', text, flags=re.IGNORECASE).strip()
            cleaned_reference = re.sub(r'\bname\b', '', SEC, flags=re.IGNORECASE).strip()

            extracted_name = cleaned_text.upper()
            reference_name = cleaned_reference.upper()

            # Calculate match ratio
            match_ratio = difflib.SequenceMatcher(None, extracted_name, reference_name).ratio()
            sec_match_percentage = round(match_ratio * 100, 2)
            print(f"Match Percentage with SEC: {sec_match_percentage}%")
    
    return {
                'CorporateName_match': corporateName_match_percentage,
                'TIN_match': tin_match_percentage,
                'sec_match': sec_match_percentage,
                'date_match': date_match_percentage
            }

pdf_path = "C:/Users/PLPASIG/Downloads/GIS_NON-STOCK(edited).pdf"

images = pdf_to_images(pdf_path)
first_page_img = images[0]

regions = {
    "Corporate Name": (411, 814, 682, 71), 
    "SEC Registration": (411, 997, 674, 98),
    "TIN": (1101, 1051, 502, 44),
    "Date": (840,114,218,36)
}

extracted_data = extract_with_bboxes(first_page_img, regions)

print("\nExtracted Information:")
for key, value in extracted_data.items():
    print(f"{key}: {value}")