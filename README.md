# Email Spam AI Visual Security & Behavior Analysis System Classifier

## 1. Project Title
**Email Spam AI Visual Security & Behavior Analysis System Classifier**

## 2. Team Members' Names
- Mahmoud Mohamed

## 3. Introduction / Problem Definition
Email spam has evolved beyond plain text messages to include images, embedded text, misleading visual cues, and behavioral manipulation techniques. Traditional spam filters that rely only on textual analysis often fail to detect image-based spam or visually deceptive content. This project addresses this limitation by designing an integrated AI system capable of analyzing **email text, images, and extracted text from images (OCR)** to make robust spam detection decisions.

## 4. Project Importance & Motivation
Spam emails pose serious risks including phishing, malware distribution, financial fraud, and privacy breaches. Modern attackers increasingly use images to bypass text-based filters. This project is motivated by the need for a **multi-modal spam detection system** that combines Natural Language Processing (NLP), Computer Vision (CV), and behavioral decision logic to enhance detection accuracy and reliability.

## 5. AI / Machine Learning Algorithms Used
The system integrates multiple AI models:

### Text Analysis
- **BERT (Bidirectional Encoder Representations from Transformers)** – Primary model for email text classification.
- **Traditional ML Model (Backup)**:
  - TfidfVectorizer
  - SGDClassifier

### Image Analysis
- **EfficientNet-B3** – Deep CNN model for classifying email images as spam or ham.

### OCR
- **Tesseract OCR** – Used to extract embedded text from images for further NLP analysis.

## 6. Methodology (Step-by-Step)
1. User uploads an email (text or image).
2. If input is text:
   - Text is analyzed using BERT.
   - If confidence < 0.9, the ML backup model participates in voting.
3. If input is an image:
   - Image is classified using EfficientNet-B3.
   - OCR extracts text from the image.
   - Extracted text is analyzed using BERT.
4. Decision Engine combines results:
   - Spam + Spam → **Spam**
   - Spam + Ham → **Suspicious**
   - Ham + Ham → **Ham**
5. Final classification is displayed via GUI.

## 7. Dataset Description & Data Preprocessing
### Text Dataset
- Enron Spam Dataset
- Preprocessing steps:
  - Lowercasing
  - Stopword removal
  - Tokenization
  - TF-IDF vectorization (for ML model)

### Image Dataset
- Spam/Ham email images
- Preprocessing steps:
  - Resize to 300x300
  - Normalization
  - Data augmentation (during training)

## 8. Model Training Procedure
### Text Models
- BERT fine-tuned using labeled email text data.
- SGDClassifier trained using TF-IDF features.

### Image Model
- EfficientNet-B3 trained on labeled email images.
- Cross-entropy loss and Adam optimizer used.

## 9. Explanation of the Main Code Components
- **text_inference.py**: Handles BERT and ML-based text classification.
- **image_inference.py**: Loads EfficientNet-B3 and performs image inference.
- **ocr_pipeline.py**: Extracts text from images using Tesseract OCR.
- **decision_engine.py**: Implements fusion logic and behavioral analysis.
- **app.py**: Streamlit-based GUI for user interaction.

## 10. Results & Evaluation Metrics

### Text-Based Spam Classification (Two Models)

**Model 1: TF-IDF + Logistic Regression (SGDClassifier)**  
- Model Type: Classical Machine Learning  
- Saved as: `ml_model.pkl` (contains both TF-IDF Vectorizer and SGDClassifier)

**Performance:**
- Accuracy: **0.9838**

| Class | Precision | Recall | F1-score | Support |
|------|-----------|--------|----------|---------|
| Ham (0) | 1.00 | 0.97 | 0.98 | 3309 |
| Spam (1) | 0.97 | 1.00 | 0.98 | 3435 |
| **Overall Accuracy** | | | **0.98** | 6744 |
| Macro Avg | 0.98 | 0.98 | 0.98 | 6744 |
| Weighted Avg | 0.98 | 0.98 | 0.98 | 6744 |

This model provides fast inference and high accuracy, making it suitable for real-time spam detection.

---

**Model 2: Transformer-based Model (DistilBERT – `distilbert-base-uncased`)**

**Evaluation Results:**
- Accuracy: **0.9936**
- Precision: **0.9953**
- Recall: **0.9921**
- Evaluation Loss: **0.0306**
- Epochs: ~2.7

DistilBERT achieved superior performance due to its deep contextual understanding of language, outperforming the classical ML model at the cost of higher computational requirements.

---

### Image-Based Spam Classification (Computer Vision)

**Model:** EfficientNet-B3  
**Pretrained Weights:** ImageNet (`IMAGENET1K_V1`)

**Performance:**

| Class | Precision | Recall | F1-score | Support |
|------|-----------|--------|----------|---------|
| Ham | 0.80 | 0.69 | 0.74 | 74 |
| Spam | 0.73 | 0.82 | 0.77 | 74 |
| **Overall Accuracy** | | | **0.76** | 148 |
| Macro Avg | 0.76 | 0.76 | 0.76 | 148 |
| Weighted Avg | 0.76 | 0.76 | 0.76 | 148 |

**Confusion Matrix:**
```
[[51 23]
 [13 61]]
```

---

### OCR Module

**OCR Engine Used:** Tesseract OCR  
The OCR component extracts textual content from email images before passing the text to the spam classification models. This enables detection of spam attempts embedded within images, improving system robustness.

**Conclusion & Future Work**
This project demonstrates the effectiveness of multi-modal spam detection using AI. By combining NLP, CV, and OCR with intelligent decision logic, the system outperforms traditional spam filters.

**Future Work:**
- Multilingual OCR and NLP support
- Real-time email server integration
- Explainable AI (XAI) visualizations

## 11.⚠️ Known Limitations & Challenges

- The image-based spam detection performance is lower compared to text-based models.
- OCR-extracted text differs significantly from formal email text.
- NLP models (BERT & ML) were trained primarily on structured email datasets, not noisy OCR-generated text.
- The image spam dataset used was relatively small and lacked diversity, affecting generalization.
- OCR errors can propagate into the text classification stage.

These limitations reflect **real-world AI challenges related to dataset quality and domain mismatch**, rather than issues in model architecture or system design.

### ⚠️ Model Files Availability Notice:

Due to storage limitations and privacy considerations, the trained model files and full datasets are not publicly guaranteed to be included in this repository.
If you need access to the trained models, datasets, or additional resources related to this project, please feel free to contact me directly via my social media platforms. I will be happy to share them upon request.

## 12.Installation
```bash
pip install -r requirements.txt
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
streamlit run app.py
```
## 13. Workflow Diagram

![Workflow][def]

[def]: assert/workflow.png

## 14. References
- Enron Email Dataset
- HuggingFace Transformers Documentation
- Tesseract OCR Documentation
- EfficientNet Research Paper

## License
This project is licensed under the MIT License and was developed as part of an academic AI course.
