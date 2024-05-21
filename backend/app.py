# app.py
import os
from flask import Flask, request, jsonify
import tensorflow as tf
from flask_cors import CORS
import numpy as np
import base64
from PIL import Image
from io import BytesIO
from logging.handlers import RotatingFileHandler
from werkzeug.utils import secure_filename
import sys
#My modules
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
app.config['UPLOAD_FOLDER'] = '/tmp'

# Load the model
#path = "/home/pabloarga/Results/2024-05-08 03.07.29/model2024-05-08 03.07.29.keras"  #BESTTTTT
path = "/home/pabloarga/Results/2024-05-09 09.50.39/model2024-05-09 09.50.39.keras"  
model = tf.keras.models.load_model(path, safe_mode=False, compile=False)

faceExtractor = FaceExtractorMultithread() 

def encondeBase64(images):
    base64_images = []
    for i,img in enumerate(images): 
        pil_img = Image.fromarray(img)
        buff = BytesIO()
        pil_img.save(buff, format="PNG")
        img_str = base64.b64encode(buff.getvalue()).decode("utf-8")
        base64_images.append(img_str)
    
    return base64_images

@app.route('/api/predict', methods=['POST'])
def predict():
    app.logger.info('Request received for predict')
    if 'video' not in request.files:
        return jsonify({'error': 'No video uploaded'}), 400
    
    video_file = request.files['video']
    video_name = secure_filename(video_file.filename)
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_name)
    video_file.save(video_path)

    # Process the video
    videoFrames = np.array([])
    processedFrames = np.array([])
    videoFrames, processedFrames = faceExtractor.process_video_to_predict(video_path)    

    # Convert frames to base64
    videoFrames64 = encondeBase64(videoFrames)
    processedFrames64 = encondeBase64(processedFrames)

    # Make predictions
    predictions = model.predict(np.stack(processedFrames, axis=0))
    predictions = [float(value) for value in predictions]
    mean = np.mean(predictions)
    var = np.var(predictions)
    range = np.max(predictions) - np.min(predictions)

    return jsonify({
        'predictions': {
            'id': video_name,
            'data': [{'x': index, 'y': value} for index, value in enumerate(predictions)]
        },
        'mean': mean,
        'var':var,
        
        'nFrames': len(predictions),
        'videoFrames': videoFrames64,
        'processedFrames': processedFrames64
    }), 200

if __name__ == '__main__':
    app.run(debug=False)
