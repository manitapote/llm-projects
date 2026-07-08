from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import BertForSequenceClassification, BertTokenizer

app = FastAPI(title="Hate Speech Detection API")

# Point to your exact model path
MODEL_PATH = "outputs/bert_hatespeech/search_20260412_182502/best_model"

print(f"Loading model from {MODEL_PATH}...")
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()
print("Model loaded successfully!")

# Adjust labels to match your training
LABELS = {0: "not_hate_speech", 1: "hate_speech"}

class TextInput(BaseModel):
    text: str

@app.get("/")
def health_check():
    return {"status": "running", "model_path": MODEL_PATH}

@app.post("/predict")
def predict(input: TextInput):
    inputs = tokenizer(
        input.text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        predicted = torch.argmax(probs, dim=1).item()
        confidence = probs[0][predicted].item()
    
    return {
        "text": input.text,
        "label": LABELS[predicted],
        "confidence": round(confidence, 4)
    }