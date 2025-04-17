from flask import Flask, request, jsonify
import librosa
import numpy as np
import soundfile as sf
import joblib
from pydub import AudioSegment
import os

app = Flask(__name__)

# Load the pre-trained model and scaler
scaler = joblib.load('feature_scaler.pkl')
clf = joblib.load('isolation_forest_model.pkl')

def extract_features(audio_file):
    y, sr = librosa.load(audio_file, sr=None)
    
    speech_rate = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, hop_length=512)
    hesitation_count = len(onset_frames)
    
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_values = pitches[pitches > 0]
    pitch_std = np.std(pitch_values)
    pitch_mean = np.mean(pitch_values)
    
    return speech_rate, hesitation_count, pitch_std, pitch_mean

@app.route('/')
def home():
    return "Welcome to the Dementia Detection API"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    os.makedirs('temp_audio', exist_ok=True)

    # Save the uploaded file
    original_path = f"temp_audio/{file.filename}"
    file.save(original_path)
    
    converted_path = original_path
    if file.filename.endswith('.mp4'):
        try:
            audio_segment = AudioSegment.from_file(original_path, format="mp4")
            converted_path = original_path.replace('.mp4', '.wav')
            audio_segment.export(converted_path, format="wav")
        except Exception as e:
            return jsonify({"error": f"Audio conversion failed: {str(e)}"}), 500

    try:
        # Feature extraction
        features = extract_features(converted_path)
        features = np.array([features])
        scaled = scaler.transform(features)
        prediction = clf.predict(scaled)

        try:
            probability = clf.predict_proba(scaled)[0].tolist()
        except AttributeError:
            probability = None

        label = "At Risk" if prediction[0] == -1 else "Normal"

        return jsonify({
            "prediction": label,
            "raw_label": int(prediction[0]),
            "probability": probability
        })

    except Exception as e:
        return jsonify({"error": f"Feature extraction failed: {str(e)}"}), 500

    finally:
        # Cleanup temporary files
        for path in {original_path, converted_path}:
            if os.path.exists(path):
                os.remove(path)

if __name__ == '__main__':
    app.run(debug=True)
