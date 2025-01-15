import cv2 as cv
from deepface import DeepFace
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import torch
import threading
import google.generativeai as genai
import os
from dotenv import load_dotenv
from collections import deque
import time
import matplotlib.pyplot as plt

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)  
gen_model = genai.GenerativeModel("gemini-1.5-flash")  

# Load EmoRoBERTa model and tokenizer
MODEL = "arpanghoshal/EmoRoBerta"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
hf_model = AutoModelForSequenceClassification.from_pretrained(MODEL, from_tf=True)

# initialize a deque with a maximum length of 5 for storing emotions
emotion_queue = deque(maxlen=5)

# webcam-based emotion recognition
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

# threaded emotion analysis
def analyze_emotion_threaded(img):
    try:
        predictions = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
        dominant_emotion = predictions[0]['dominant_emotion']
        
        # check if the emotion is unique and update the queue
        if emotion_queue and emotion_queue[-1] != dominant_emotion:
            emotion_queue.append(dominant_emotion)
        elif not emotion_queue:
            emotion_queue.append(dominant_emotion)
    except Exception as e:
        pass


def webcam_emotion_recognition():
    haar_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
    feed = cv.VideoCapture(0)

    if not feed.isOpened():
        print("Cannot open webcam")
        return None

    print("Press 'd' to exit webcam.")
    
    while True:
        isTrue, img = feed.read()
        if not isTrue:
            break

        img = draw_boundary(img, haar_cascade, scaleFactor=1.1, minNeighbors=10, color=(255, 0, 0))
        cv.imshow('Webcam Emotion Recognition', img)

        key = cv.waitKey(20) & 0xFF
        if key == ord('d'):
            print("Exiting webcam.")
            break

        threading.Thread(target=analyze_emotion_threaded, args=(img,)).start()

    feed.release()
    cv.destroyAllWindows()

    return emotion_queue

# text Emotion Analysis
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

def bar_chart(emotion_labels, probs):
    sorted_probs, sorted_emotions = zip(*sorted(zip(probs, emotion_labels), reverse=True))
    plt.figure(figsize=(10, 6))
    plt.barh(sorted_emotions, sorted_probs, color=plt.cm.Paired.colors[:len(sorted_emotions)])
    plt.xlabel('Probability')
    plt.title('Emotion Distribution', fontsize=14)
    plt.gca().invert_yaxis()
    plt.show()


def text_emotion_analysis():
    user_input = input("Enter your thought: ")
    encoded_input = preprocess_input(user_input)
    dominant_emotion, probs, emotion_labels = analyze_input(encoded_input)
    
    # update emotion queue with the detected emotion
    if emotion_queue and emotion_queue[-1] != dominant_emotion:
        emotion_queue.append(dominant_emotion)
    elif not emotion_queue:
        emotion_queue.append(dominant_emotion)
    bar_chart(emotion_labels, probs)
    return dominant_emotion, user_input  

# generate response based on emotion
def generate_personalized_response(emotion, text=""):
    prompt = f"User is feeling {emotion}. Based on their emotion, suggest some activities or advice to help them feel better."
    if text:  
        prompt += f" Additionally, the user said: {text}"
    
    response = gen_model.generate_content(contents=[{"text": prompt}])
    print(f"Personalized Advice: {response.text}")
    return response.text

def unified_emotion_detection():
    print("Choose Input Mode:")
    print("1. Webcam (Face Detection + Emotion Recognition)")
    print("2. Text Input (Emotion Detection)")

    mode = input("Enter 1 or 2: ")

    if mode == "1":
        emotion_queue = webcam_emotion_recognition()
        if emotion_queue:
            print("Emotions detected from webcam:", list(emotion_queue))
            generate_personalized_response(emotion_queue[-1])
        else:
            print("No emotion detected from webcam.")
    
    elif mode == "2":
        dominant_emotion, user_input = text_emotion_analysis()
        print(f"Dominant Emotion from Text: {dominant_emotion}")
        generate_personalized_response(dominant_emotion, user_input)
    else:
        print("Invalid input. Please enter 1 or 2.")

if __name__ == "__main__":
    unified_emotion_detection()
