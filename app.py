import os
from flask import Flask, request, jsonify, render_template
import joblib
import librosa
import numpy as np

app = Flask(__name__)

# Load the trained model
model_filename = r'C:\Benish\speech_emotion_detection_model.pkl'
model = joblib.load(model_filename)
print(f"Model loaded from {model_filename}")

# Function to extract features from audio files
def extract_features(file_name):
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        return mfccs_scaled
    except Exception as e:
        print(f"Error encountered while parsing file: {file_name}")
        print(e)
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)
        
        features = extract_features(file_path)
        if features is not None:
            features = features.reshape(1, -1)  # Reshape for a single sample
            prediction = model.predict(features)
            return jsonify({'emotion': prediction[0]})
        else:
            return jsonify({'error': 'Failed to extract features from the audio file'})

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)