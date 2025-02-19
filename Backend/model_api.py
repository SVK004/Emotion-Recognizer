from flask import Flask, request, jsonify
import numpy as np
import librosa
from pathlib import Path
import os
import sys
from flask_cors import CORS


parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))


from rough import get_accuracy

app = Flask(__name__)
CORS(app)

# Directory to save the uploaded audio files temporarily
UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load trained model parameters
W1 = np.loadtxt("../w1.txt")
b1 = np.loadtxt("../b1.txt")
W2 = np.loadtxt("../w2.txt")
b2 = np.loadtxt("../b2.txt")
b1 = b1[:, :1]
b2 = b2[:, :1]

@app.route('/process-audio', methods=['POST'])
def process_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_file.filename)
    audio_file.save(file_path)

    try:
        accuracy = get_accuracy(W1, b1, W2, b2, file_path)
        return jsonify({"accuracy": accuracy})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)




if __name__ == "__main__":
    app.run(debug=True)
