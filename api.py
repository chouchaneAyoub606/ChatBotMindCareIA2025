from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import json
import random
import os
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
import uuid  # for unique message IDs
import requests

# ✅ Define constants first
MODEL_URL = "https://huggingface.co/ayoubbob606/ChatBotMindCareIA/resolve/main/model.safetensors"
MODEL_PATH = "bert_model/model.safetensors"

# ✅ Download the model if it doesn't exist or is corrupted
def download_model():
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 10000000:
        print("📦 Downloading model from Hugging Face...")
        response = requests.get(MODEL_URL, allow_redirects=True)
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)
        print("✅ Model downloaded.")
    else:
        print("✅ Model already exists and looks fine.")

# ✅ Call before loading the model
download_model()

# ✅ Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Initialize Firebase
cred = credentials.Certificate("firebase_key.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# ✅ Create FastAPI app
app = FastAPI()

# ✅ Define model paths
model_path = "bert_model"
intents_path = "expanded_intents_mental_health.json"
label_map_path = os.path.join(model_path, "label_map.json")

# ✅ Load model and tokenizer
model = BertForSequenceClassification.from_pretrained(model_path).to(device)
tokenizer = BertTokenizer.from_pretrained(model_path)

# ✅ Load label map
with open(label_map_path, "r", encoding="utf-8") as f:
    id2tag = json.load(f)

# ✅ Load intents
with open(intents_path, "r", encoding="utf-8") as f:
    intents = json.load(f)

# ✅ Save conversation to Firebase
def save_message(chat_id, sender, message):
    doc_ref = db.collection("chatbot").document(chat_id).collection("messages").document(str(uuid.uuid4()))
    doc_ref.set({
        "sender": sender,
        "message": message,
        "timestamp": datetime.utcnow().isoformat()
    })

# ✅ Get a random response for predicted tag
def get_response(intent_tag):
    for intent in intents['intents']:
        if intent['tag'] == intent_tag:
            return random.choice(intent['responses'])

# ✅ Predict tag and confidence
def predict_class(text):
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding='max_length', max_length=32).to(device)
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        top_pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][top_pred].item()
        return id2tag[str(top_pred)], confidence

# ✅ Request body schema
class ChatRequest(BaseModel):
    message: str

# ✅ Chat endpoint
@app.post("/chat")
def chat(req: ChatRequest):
    user_input = req.message
    tag, confidence = predict_class(user_input)
    if confidence > 0.3:
        response = get_response(tag)
    else:
        response = "I'm sorry, I didn't quite understand that. Can you rephrase?"

    chat_id = "chatid1"  # ✅ Replace with dynamic user/session ID later

    save_message(chat_id, "user", user_input)
    save_message(chat_id, "bot", response)

    return {
        "tag": tag,
        "confidence": round(confidence, 2),
        "response": response
    }

# ✅ Optional health check route
@app.get("/")
def root():
    return {"status": "MindCare Chatbot API is running"}
