# app.py
from flask import Flask, request, jsonify
import tensorflow as tf
import cv2
import numpy as np
#My modules
from training.FaceReconModule import FaceExtractorMultithread

import sys
sys.path.append("..")

app = Flask(__name__)

# Load your TensorFlow model
model = tf.keras.models.load_model('/home/pabloarga/Results/2024-03-04 14.39.53/model2024-03-04 14.39.53.keras',safe_mode=False,compile=False)
faceExtractor = FaceExtractorMultithread() 
video_path = '/tmp/video.mp4'

@app.route('/predict', methods=['POST'])
def predict():
    if 'video' not in request.files:
        return jsonify({'error': 'No video uploaded'}), 400
    
    video_file = request.files['video']
    video_file.save(video_path)
    # Process the video
    frames = faceExtractor.process_video_to_predict(video_path)

    # Make predictions
    predictions = model.predict(np.stack(frames, axis=0))

     # Convert predictions to regular Python list and convert numpy.float32 to Python float
    predictions = [float(value) for value in predictions]

    mean = np.mean(predictions)

    return jsonify({'predictions': predictions , 'mean' : mean, "nFrames" : len(predictions)}), 200

if __name__ == '__main__':
    app.run(debug=True)
