from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np
import cv2
from PIL import Image
import io

from src.model import DETR

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all (for development)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Correct label order (from your config)
labels = ["hello", "iloveyou", "thankyou"]

# DETR internally adds background → so num_classes = 3
model = DETR(num_classes=3)

# Load pretrained weights
state_dict = torch.load("pretrained/4426_model.pt", map_location="cpu")
model.load_state_dict(state_dict)

model.eval()

@app.get("/")
def home():
    return {"message": "Sign Language API Running 🚀"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        # Convert image
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img = np.array(image)

        # Preprocess
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        img = np.transpose(img, (2, 0, 1))

        tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            outputs = model(tensor)

        # 🔥 DETR correct handling
        logits = outputs["pred_logits"]   # [1, queries, classes+1]
        logits = logits[0]                # [queries, classes+1]

        probs = torch.softmax(logits, dim=-1)

        # Ignore background (last class)
        scores, classes = torch.max(probs[:, :-1], dim=-1)

        # Best detection
        best_idx = torch.argmax(scores)

        pred_class = classes[best_idx].item()
        confidence = scores[best_idx].item()

        return {
            "prediction": labels[pred_class],
            "confidence": round(confidence, 2)
        }

    except Exception as e:
        return {"error": str(e)}