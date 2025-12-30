import cv2
import pytesseract
import re

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def preprocess_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]
    return thresh


def clean_text(text):
    text = re.sub(r"http\S+", " URL ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_text(image_path):
    img = preprocess_image(image_path)
    raw_text = pytesseract.image_to_string(img, lang="eng")
    return clean_text(raw_text)
