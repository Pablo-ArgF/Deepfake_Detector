import os
from flask import Flask, request, jsonify
import tensorflow as tf
from flask_cors import CORS
import numpy as np
from PIL import Image
from logging.handlers import RotatingFileHandler
from werkzeug.utils import secure_filename
import sys
import cv2
# My modules
sys.path.append("..")
from training.DataProcessing.FaceReconModule import FaceExtractorMultithread

import logging
from logging.handlers import RotatingFileHandler
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configure logging to a file with rotation
handler = RotatingFileHandler('app.log', maxBytes=10000, backupCount=1)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
app.logger.addHandler(handler)

CORS(app, origins='*', methods=['GET', 'POST'], allow_headers=['Content-Type'])
# Increase maximum content length to 4Gb
app.config['MAX_CONTENT_LENGTH'] = 4 * 1024 * 1024 *1024  # 4Gb
app.config['VIDEO_UPLOAD_FOLDER'] = '/tmp'
app.config['UPLOAD_FOLDER'] = '/home/pabloarga/Deepfake_Detector/frontend/build/results'
app.config['UPLOAD_FOLDER_REF'] = './results' #REference for the frontend src
app.config['SELECTED_MODEL'] = '2024-06-05 09.29.28'

# Load the model
path = f"/home/pabloarga/Results/{app.config['SELECTED_MODEL']}/model{app.config['SELECTED_MODEL']}.keras"  
model = tf.keras.models.load_model(path, safe_mode=False, compile=False)

faceExtractor = FaceExtractorMultithread() 

def save_images(frames, video_name):
    """
    Guarda las imágenes en el directorio de carga.
    """
    image_files = []
    for i, frame in enumerate(frames):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{video_name}_frame_{i}.png')
        cv2.imwrite(file_path, frame)
        image_files.append(os.path.join(app.config['UPLOAD_FOLDER_REF'], f'{video_name}_frame_{i}.png'))
    return image_files
    
def remove_all_files(folder_path):
    # Iterate over all the files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            # Remove the file
            os.remove(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

@app.route('/api/predict', methods=['POST'])
def predict():
    app.logger.info('Request received for predict')
    if 'video' not in request.files:
        return jsonify({'error': 'No video uploaded'}), 400

    #Remove previous frame files
    remove_all_files(app.config['UPLOAD_FOLDER'])
    
    video_file = request.files['video']
    video_name = secure_filename(video_file.filename)
    video_path = os.path.join(app.config['VIDEO_UPLOAD_FOLDER'], video_name)
    video_file.save(video_path)

    # Process the video
    videoFrames, processedFrames = faceExtractor.process_video_to_predict(video_path)    

    # Make predictions
    predictions = model.predict(np.stack(processedFrames, axis=0))
    predictions = [float(value) for value in predictions]
    mean = np.mean(predictions)
    var = np.var(predictions)
    maxVal = np.max(predictions)
    minVal = np.min(predictions)
    range_ = maxVal - minVal

    # Save the images and get their paths
    video_frame_files = save_images(videoFrames, video_name)
    processed_frame_files = save_images(processedFrames, f'{video_name}_processed')

    return jsonify({
        'predictions': {
            'id': video_name,
            'data': [{'x': index, 'y': value} for index, value in enumerate(predictions)]
        },
        'mean': mean,
        'var':var,
        'max':maxVal,
        'min':minVal,
        'range':range_,
        'nFrames': len(predictions),
        'videoFrames': video_frame_files,
        'processedFrames': processed_frame_files
    }), 200

if __name__ == '__main__':
    app.run(debug=False)
