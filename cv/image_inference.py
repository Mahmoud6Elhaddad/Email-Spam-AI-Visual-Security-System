import torch
import timm
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR = Path(__file__).resolve().parent
IMAGE_MODEL_PATH = BASE_DIR / "best_image_model.pth"

LABEL_MAP = {0: "ham", 1: "spam"}

image_transforms = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_image_model.pth"

LABEL_MAP = {0: "ham", 1: "spam"}

transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ✅ SAME architecture used in training
model = efficientnet_b3(weights=None)

# Replace classifier head
model.classifier[1] = torch.nn.Linear(
    model.classifier[1].in_features,
    2
)

state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)

model.to(DEVICE)
model.eval()


def image_predict(image_path):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

    pred = int(np.argmax(probs))
    confidence = float(probs[pred])

    return LABEL_MAP[pred], confidence


state = torch.load(IMAGE_MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state)
model.to(DEVICE)
model.eval()


def image_predict(image_path: str):
    image = Image.open(image_path).convert("RGB")
    tensor = image_transforms(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

    pred = int(np.argmax(probs))
    confidence = float(probs[pred])

    return LABEL_MAP[pred], confidence
