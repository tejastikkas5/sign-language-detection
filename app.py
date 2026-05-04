from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np
import cv2
from PIL import Image
import io
import os
import requests

from src.model import DETR

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

labels = ["hello", "iloveyou", "thankyou"]

MODEL_PATH = "pretrained/4426_model.pt"

model = None  # 👈 initially empty


# 🚀 Load model AFTER server starts
@app.on_event("startup")
def load_model():
    global model
    print("🚀 Starting app...")

    os.makedirs("pretrained", exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        print("📥 Downloading model...")
        url = "https://huggingface.co/tejas55/sign-language-model/resolve/main/4426_model.pt"
        r = requests.get(url)
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)

    print("📦 Loading model...")
    model = DETR(num_classes=3)
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    print("✅ Model ready!")


@app.get("/")
def home():
    return {"message": "Sign Language API Running 🚀"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img = np.array(image)

        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        img = np.transpose(img, (2, 0, 1))

        tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            outputs = model(tensor)

        logits = outputs["pred_logits"][0]
        probs = torch.softmax(logits, dim=-1)

        scores, classes = torch.max(probs[:, :-1], dim=-1)
        best_idx = torch.argmax(scores)

        pred_class = classes[best_idx].item()
        confidence = scores[best_idx].item()

        return {
            "prediction": labels[pred_class],
            "confidence": round(confidence, 2)
        }

    except Exception as e:
        return {"error": str(e)}
