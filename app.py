"""
Author: Hamza Bayraktepe

Description: This Flask application performs emotion recognition on uploaded audio and video files. The audio emotion recognition utilizes the Wav2Vec2 model, while the video emotion recognition uses a pre-trained facial emotion detection model.
"""

import os
import numpy as np
from flask import Flask, render_template, request, jsonify
import cv2
import librosa
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from transformers import pipeline

# Initialize Flask application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_AUDIO_EXTENSIONS'] = {'wav', 'ogg'}
app.config['ALLOWED_VIDEO_EXTENSIONS'] = {'mp4'}

# Load models
audio_model = pipeline("audio-classification", model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
video_model = load_model("facialemotionmodel.h5")

# Define emotion labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Load face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def allowed_audio_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_AUDIO_EXTENSIONS']


def allowed_video_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_VIDEO_EXTENSIONS']


def extract_audio_features(file_path):
    audio_data, sample_rate = librosa.load(file_path, sr=16000)
    return audio_data


def extract_video_features(file_path):
    cap = cv2.VideoCapture(file_path)
    success, frame = cap.read()
    if success:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
        for (x, y, w, h) in faces:
            roi_gray = gray_frame[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            roi_gray = roi_gray / 255.0
            roi_gray = np.expand_dims(roi_gray, axis=-1)
            return roi_gray
    return None


@app.route('/', methods=["GET", "POST"])
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'audio' not in request.files or 'video' not in request.files:
        return jsonify({"error": "No audio or video file part"})

    audio_file = request.files['audio']
    video_file = request.files['video']

    if audio_file.filename == '' or video_file.filename == '':
        return jsonify({"error": "No selected audio or video file"})

    if audio_file and allowed_audio_file(audio_file.filename) and video_file and allowed_video_file(
            video_file.filename):
        audio_file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(audio_file.filename))
        video_file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(video_file.filename))

        audio_file.save(audio_file_path)
        video_file.save(video_file_path)

        # Audio emotion recognition
        audio_features = extract_audio_features(audio_file_path)
        audio_prediction = audio_model(audio_features)
        audio_emotion = audio_prediction[0]['label']

        # Video emotion recognition
        video_features = extract_video_features(video_file_path)
        if video_features is None:
            return jsonify({"error": "No face detected in the video frame"})

        video_features = np.expand_dims(video_features, axis=0)
        video_prediction = video_model.predict(video_features)
        video_maxindex = int(np.argmax(video_prediction))
        video_emotion = emotion_labels[video_maxindex]

        return jsonify({"audio_emotion": audio_emotion, "video_emotion": video_emotion})

    return jsonify({"error": "File type not allowed"})


if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
