from text.text_inference import bert_predict, ml_predict
from cv.image_inference import image_predict
from OCR.ocr_pipeline import extract_text


def text_decision(text: str):
    bert_label, bert_conf = bert_predict(text)

    if bert_conf > 0.9:
        return bert_label, bert_conf, "BERT"

    ml_label, ml_conf = ml_predict(text)

    final = bert_label if bert_label == ml_label else "spam"
    confidence = max(bert_conf, ml_conf)

    return final, confidence, "BERT + ML Voting"


def image_decision(image_path: str):
    img_label, img_conf = image_predict(image_path)
    ocr_text = extract_text(image_path)

    if len(ocr_text) < 10:
        return img_label, img_conf, "Image Only"

    text_label, text_conf, _ = text_decision(ocr_text)

    if img_label == "spam" and text_label == "spam":
        return "spam", max(img_conf, text_conf), "Image + OCR (Confirmed)"

    if img_label != text_label:
        return "suspicious", max(img_conf, text_conf), "Behavior Conflict"

    return "ham", max(img_conf, text_conf), "Image + OCR"
