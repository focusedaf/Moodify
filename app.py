from flask import Flask, render_template, Response, jsonify, request
import threading
import cv2 as cv
from combined import webcam_emotion_recognition, emotion_queue,analyze_input, generate_personalized_response, preprocess_input, generate_bar_chart
import queue

app = Flask(__name__)

webcam_thread_started = False
video_capture = None
feed_lock = threading.Lock()
unique_emotions = []
MAX_EMOTIONS = 5  

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/webcam', methods=['GET'])
def webcam():
    return render_template('webcam.html')

@app.route('/start_webcam', methods=['POST'])
def start_webcam():
    global webcam_thread, webcam_thread_started, video_capture
    unique_emotions.clear()
    emotion_queue.clear()

    if not webcam_thread_started:
        video_capture = cv.VideoCapture(0)
        webcam_thread = threading.Thread(target=generate_video_feed, daemon=True)
        webcam_thread.start()
        webcam_thread_started = True
        return jsonify({"status": "Webcam started successfully."})
    else:
        return jsonify({"status": "Webcam is already running."})

@app.route('/stop_webcam', methods=['POST'])
def stop_webcam():
    global webcam_thread_started, video_capture
    if webcam_thread_started:
        with feed_lock:
            if video_capture:
                video_capture.release()
                video_capture = None
        webcam_thread_started = False
        return jsonify({"status": "Webcam stopped successfully."})
    else:
        return jsonify({"status": "Webcam is not running."})

def generate_video_feed():
    global video_capture, unique_emotions
    while True:
        if video_capture and video_capture.isOpened():
            ret, frame = video_capture.read()
            if not ret:
                break
            emotion, frame = webcam_emotion_recognition(frame)

            if emotion:
                if emotion not in unique_emotions:
                    if len(unique_emotions) >= MAX_EMOTIONS:
                        unique_emotions.pop(0)
                    unique_emotions.append(emotion)

            
            ret, jpeg = cv.imencode('.jpg', frame)
            if ret:
                frame = jpeg.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_video_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_emotion_response', methods=['POST'])
def get_emotion_response():
    global unique_emotions
    if unique_emotions:
        emotion = unique_emotions[-1] 
        ai_response = generate_personalized_response(emotion, emotion)  
        return jsonify({"emotion": emotion, "response": ai_response})
    return jsonify({"emotion": "None", "response": "No emotion detected."})



@app.route('/analyze_image', methods=['POST'])
def analyze_image():
    try:
        image_data = request.form['image']
        image_data = image_data.split(",")[1]  
        image = base64.b64decode(image_data)
        np_img = np.frombuffer(image, np.uint8)
        frame = cv.imdecode(np_img, cv.IMREAD_COLOR)

    
        emotion, frame = webcam_emotion_recognition(frame)
        response = generate_personalized_response(emotion)

        return jsonify({
            "dominant_emotion": emotion,
            "response": response
        })
    except Exception as e:
        print(f"Error analyzing image: {e}")
        return jsonify({"error": "An error occurred during analysis."}), 500



@app.route('/text')
def text():
    return render_template('text.html')

@app.route('/analyze_text', methods=['POST'])
def analyze_text():
    try:
        user_input = request.json.get("text", "")
        if not user_input:
            return jsonify({"error": "Text input is required"}), 400
        encoded_input = preprocess_input(user_input)
        dominant_emotion, probs, emotion_labels = analyze_input(encoded_input)
        ai_response = generate_personalized_response(dominant_emotion, user_input, source="text")
        bar_chart_img = generate_bar_chart(emotion_labels, probs)
        return jsonify({
            "dominant_emotion": dominant_emotion,
            "response": ai_response,
            "bar_chart_img": bar_chart_img
        })
    except Exception as e:
        print(f"Error during analysis: {e}")
        return jsonify({"error": "An error occurred during analysis."}), 500

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)