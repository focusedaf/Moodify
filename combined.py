import io  
import cv2 as cv
from deepface import DeepFace
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import torch
import google.generativeai as genai
import os
from dotenv import load_dotenv
from collections import deque
import time
import matplotlib.pyplot as plt
import threading
import base64


load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)
gen_model = genai.GenerativeModel("gemini-1.5-flash")

MODEL = "arpanghoshal/EmoRoBerta"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
hf_model = AutoModelForSequenceClassification.from_pretrained(MODEL, from_tf=True)

emotion_queue = deque(maxlen=5)

def draw_boundary(img, classifier, scaleFactor, minNeighbors, color):
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces_rect = classifier.detectMultiScale(gray_img, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
    for (x, y, w, h) in faces_rect:
        cv.rectangle(img, (x, y), (x + w, y + h), color, thickness=2)
        face_roi = img[y:y + h, x:x + w]
        face_rgb = cv.cvtColor(face_roi, cv.COLOR_BGR2RGB)
        try:
            predictions = DeepFace.analyze(face_rgb, actions=['emotion'], enforce_detection=False)
            dominant_emotion = predictions[0]['dominant_emotion']
            cv.putText(img, dominant_emotion, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv.LINE_AA)
        except Exception as e:
            pass
    return img

def analyze_emotion_threaded(img):
    try:
        
        predictions = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
        dominant_emotion = predictions[0]['dominant_emotion']

        if dominant_emotion not in emotion_queue:
            emotion_queue.append(dominant_emotion)


        if len(emotion_queue) >= 5:
            # print("5 unique emotions collected, stopping emotion detection.")
            return True  

    except Exception as e:
        pass
    return False

def webcam_emotion_recognition(frame):
    haar_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
    frame = draw_boundary(frame, haar_cascade, scaleFactor=1.1, minNeighbors=10, color=(255, 0, 0))
    if analyze_emotion_threaded(frame):
        return "done", frame  
    return emotion_queue[-1] if emotion_queue else None, frame

def preprocess_input(user_input):
    return tokenizer(user_input.lower(), return_tensors="pt")

def analyze_input(encoded_input):
    with torch.no_grad():
        output = hf_model(**encoded_input)
    logits = output.logits.detach().numpy()
    probs = softmax(logits[0])
    emotion_labels = [
        "admiration", "amusement", "anger", "annoyance", "approval", "caring",
        "confusion", "curiosity", "desire", "disappointment", "disapproval",
        "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
        "joy", "love", "nervousness", "optimism", "pride", "realization", "relief",
        "remorse", "sadness", "surprise", "neutral"
    ]
    dominant_emotion = emotion_labels[probs.argmax()]
    return dominant_emotion, probs, emotion_labels

def generate_personalized_response(emotion, text="", source=""):
    prompt = f"User is feeling {emotion}. Based on their emotion, suggest some activities or advice to help them feel better."
    if source == "text":
        prompt += " This emotion was detected from the user's text input."
    if text:
        prompt += f" Additionally, the user said: {text}"
    try:
        response = gen_model.generate_content(contents=[{"text": prompt}])
        return response.text  
    except Exception as e:
        print(f"Error generating response: {e}")
        return "An error occurred while generating a response."

def generate_bar_chart(emotion_labels, probs):
    sorted_probs, sorted_emotions = zip(*sorted(zip(probs, emotion_labels), reverse=True))
    plt.figure(figsize=(10, 6))
    plt.barh(sorted_emotions, sorted_probs, color=plt.cm.Paired.colors[:len(sorted_emotions)])
    plt.xlabel('Probability')
    plt.title('Emotion Distribution', fontsize=14)
    plt.gca().invert_yaxis()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    img_data = base64.b64encode(img.read()).decode('utf-8')
    plt.close()
    return img_data  
