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

# ✅ CORS (allow frontend access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Labels (must match training config)
labels = ["hello", "iloveyou", "thankyou"]

# ✅ Model path
MODEL_PATH = "pretrained/4426_model.pt"


# 🚀 Download model if not exists
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("⬇️ Downloading model from Hugging Face...")
        os.makedirs("pretrained", exist_ok=True)

        url = "https://huggingface.co/tejas55/sign-language-model/resolve/main/4426_model.pt"

        try:
            r = requests.get(url, stream=True)
            r.raise_for_status()

            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            print("✅ Model downloaded successfully")

        except Exception as e:
            print("❌ Model download failed:", e)
            raise RuntimeError("Failed to download model")


# 🚀 Load model once at startup
download_model()

model = DETR(num_classes=3)

state_dict = torch.load(MODEL_PATH, map_location="cpu")
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

        # Inference
        with torch.no_grad():
            outputs = model(tensor)

        # DETR output
        logits = outputs["pred_logits"][0]  # [queries, classes+1]
        probs = torch.softmax(logits, dim=-1)

        # Ignore background class (last index)
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
