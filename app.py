import os
from flask import Flask, request, send_file, jsonify
from werkzeug.utils import secure_filename
import soundfile as sf
import uuid
import threading
from audio_denoiser.AudioDenoiser import AudioDenoiser
import torch
import torchaudio

app = Flask(__name__)

# Configuration for upload and processed files
# These paths are relative to the current directory (the repo root)
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed_audio'
ALLOWED_EXTENSIONS = {'wav', 'mp3'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# Create upload and processed folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# use CPU
# denoiser = AudioDenoiser()

# use GPU (else CPU)
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
denoiser = AudioDenoiser(device=device)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/denoise', methods=['POST'])
def denoise_audio():
    if 'audio_file' not in request.files:
        return jsonify({"error": "No audio_file part in the request"}), 400

    file = request.files['audio_file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        unique_id = str(uuid.uuid4())
        input_filename = f"{unique_id}_input_{filename}"
        
        output_filename = f"{unique_id}_denoised.wav" 

        input_filepath = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)
        output_filepath = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)
        
        file.save(input_filepath)

        try:
            denoiser.process_audio_file(input_filepath, output_filepath)
            os.remove(input_filepath)
            
            return send_file(output_filepath, mimetype='audio/wav', as_attachment=True, download_name=output_filename)

        except Exception as e:
            if os.path.exists(input_filepath):
                os.remove(input_filepath)
            return jsonify({"error": f"Error processing audio: {str(e)}"}), 500
    else:
        return jsonify({"error": "Invalid file type. Allowed types are .wav, .mp3"}), 400

if __name__ == '__main__':
    flask_thread = threading.Thread(target=app.run, kwargs={
        "host": "0.0.0.0",
        "port": 7861
    })
    flask_thread.daemon = True
    flask_thread.start()