# **Moodify: AI-Powered Emotion Recognition and Response**

Moodify is an AI-powered platform that recognizes emotions from video or text input and generates personalized responses based on the detected emotion. Moodify provides two interfaces: one with a **UI (Web Interface)** for webcam and real-time video interaction, and one with a **Command-Line Interface (CLI)** for text-based analysis and emotion recognition.

## **Table of Contents**
1. [Overview](#overview)
2. [Features](#features)
3. [Technologies Used](#technologies-used)
4. [Getting Started](#getting-started)
    - [UI Version (Web Interface)](#ui-version-web-interface)
    - [CLI Version](#cli-version)
5. [Running Moodify](#running-moodify)
    - [UI Version](#running-the-ui-version)
    - [CLI Version](#running-the-cli-version)
6. [Usage](#usage)
    - [UI Version](#usage-ui-version)
    - [CLI Version](#usage-cli-version)

---

## **Overview**

Moodify is an emotion recognition tool that analyzes user inputs (via webcam or text) and generates responses based on the detected emotions. It can be used for various applications, such as mood tracking, mental health support, and personalized experiences.

Moodify offers two versions:
- **UI Version (Web Interface)**: Provides a real-time interaction using a webcam to detect emotions and generate corresponding responses.
- **CLI Version**: Allows users to input text for emotion recognition and response generation directly from the command line.

---

## **Features**
- **Emotion Detection**: Recognizes emotions such as happiness, sadness, anger, surprise, etc., from webcam video or text input.
- **Personalized Response Generation**: Based on the detected emotion, the AI generates a personalized response for the user.
- **Web Interface**: Real-time emotion detection using webcam feed and display of AI-generated responses.
- **Command-Line Interface**: Analyze text input and display emotion-related insights through the command line.

---

## **Technologies Used**
- **Backend**: Python, Flask
- **AI Models**: OpenCV, DNN, GPT (for AI responses)
- **Frontend (UI Version)**: HTML, CSS, JavaScript (with TailwindCSS)
- **Emotion Recognition**: Haar Cascades, DNN, OpenCV
- **AI Response Generation**: GPT-based model (like GPT-Neo-1.3B)

---

## **Getting Started**

### **UI Version (Web Interface)**

#### Prerequisites:
1. Python (v3.7+)
2. Node.js and npm (if customizing frontend or managing dependencies)
3. Install the required Python libraries and dependencies.

You can set up the UI version with the following steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/Moodify.git
    cd moodify
    ```

2. Install Python dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Install frontend dependencies (if needed):
    ```bash
    npm install
    ```

4. Run the Flask app:
    ```bash
    python app.py
    ```

5. Open your browser and go to `http://127.0.0.1:5000/` to view the Moodify UI version.

### **CLI Version**

#### Prerequisites:
1. Python (v3.7+)
2. Install the required Python libraries and dependencies.

To set up the CLI version:

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/Moodify.git
    cd moodify
    ```

2. Install Python dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the CLI script:
    ```bash
    python cmdcombined.py
    ```

---

## **Running Moodify**

### **UI Version (Web Interface)**

Once the app is running, you can:
- **Start Webcam**: Click the "Start Webcam" button to start the webcam and begin emotion detection.
- **Stop Webcam**: Click the "Stop Webcam" button to stop the webcam and clear the results.

- **AI Response**: The detected emotion will trigger a response from the AI, which will be shown in the text area.

#### Endpoints:
- **`/start_webcam`**: Starts the webcam for real-time emotion detection.
- **`/stop_webcam`**: Stops the webcam.
- **`/video_feed`**: Streams the video feed from the webcam, with emotion detection overlays.

### **CLI Version**

For the CLI version, you will input text and get an emotion analysis followed by a personalized response.

Example:
```bash
$ python cmdcombined.py
Enter text to analyze: I'm feeling a bit down today.
Detected Emotion: Sadness
AI Response: It's okay to feel sad. Take a deep breath and give yourself some time.
```
---

## **Usage**

### **UI Version (Web Interface)**

1. Start Webcam:
- Click "Start Webcam" to begin emotion detection.
- The system will capture your face and analyze your emotions in real-time.
2. AI Response:
- The detected emotion will be displayed.
- The system will generate a corresponding response based on the emotion detected.
3. Stop Webcam:
- Click "Stop Webcam" to stop the video feed.
- The text area will be cleared, and the webcam will stop streaming.
  
### **CLI Version**

1. Text Input:
- Run the script in the terminal.
- Enter any text to get an emotion analysis and an AI response based on the detected emotion.
2. AI Response:
- The system will display the detected emotion and generate a relevant response based on the 
   input text.

