import os
from flask import Flask, request, jsonify
import tensorflow as tf
from flask_cors import CORS
import numpy as np
from PIL import Image
import base64
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

CORS(app, resources={r"/*": {"origins": "*"}})


# Increase maximum content length to 4Gb
app.config['MAX_CONTENT_LENGTH'] = 4 * 1024 * 1024 *1024  # 4Gb
app.config['VIDEO_UPLOAD_FOLDER'] = '/tmp'
app.config['UPLOAD_FOLDER'] = '/home/pabloarga/Deepfake_Detector/frontend/build/results'
app.config['UPLOAD_FOLDER_REF'] = './results' #REference for the frontend src
app.config['SELECTED_CNN_MODEL'] = '2024-06-20 08.29.00' 
app.config['SELECTED_RNN_MODEL'] = '2024-06-13 16.20.00'
app.config['RNN_MODEL_SEQUENCE_LENGTH'] = 20

# Load the cnn model
path = f"/home/pabloarga/Results/{app.config['SELECTED_CNN_MODEL']}/model{app.config['SELECTED_CNN_MODEL']}.keras"  
model = tf.keras.models.load_model(path, safe_mode=False, compile=False)

# Load the rnn model
pathSequences = f"/home/pabloarga/Results/{app.config['SELECTED_RNN_MODEL']}/model{app.config['SELECTED_RNN_MODEL']}.keras"  
modelSequences = tf.keras.models.load_model(pathSequences, safe_mode=False, compile=False)


faceExtractor = FaceExtractorMultithread() 

def image_to_base64(image_path):
    with open(image_path, 'rb') as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

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

def save_sequences(sequences, video_name):
    """
    Guarda las imagenes de las secuencias en el directorio de carga.
    """
    image_files = []
    for sequence in sequences:
        for i, frame in enumerate(sequence):
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

@app.route('/api/model/structure/cnn', methods=['GET'])
def getCNNStructure():
    app.logger.info('Request received for getCNNStructure')
    return image_to_base64(f'/home/pabloarga/Results/{app.config["SELECTED_CNN_MODEL"]}/model_structure.png')

@app.route('/api/model/structure/rnn', methods=['GET'])
def getRNNStructure():
    app.logger.info('Request received for getRNNStructure')
    return image_to_base64(f'/home/pabloarga/Results/{app.config["SELECTED_RNN_MODEL"]}/model_structure.png')

@app.route('/api/model/graphs/cnn', methods=['GET'])
def getCNNGraphs():
    app.logger.info('Request received for getCNNGraphs')
    return image_to_base64(f'/home/pabloarga/Results/{app.config["SELECTED_CNN_MODEL"]}/combined_plots.png')

@app.route('/api/model/graphs/rnn', methods=['GET'])
def getRNNGraphs():
    app.logger.info('Request received for getRNNGraphs')
    return image_to_base64(f'/home/pabloarga/Results/{app.config["SELECTED_RNN_MODEL"]}/combined_plots.png')

@app.route('/api/model/confussion/matrix/cnn', methods=['GET'])
def getCNNConfussionMatrix():
    app.logger.info('Request received for getCNNConfussionMatrix')
    return image_to_base64(f'/home/pabloarga/Results/{app.config["SELECTED_CNN_MODEL"]}/confusionMatrix_{app.config["SELECTED_CNN_MODEL"]}.png')

@app.route('/api/model/confussion/matrix/rnn', methods=['GET'])
def getRNNConfussionMatrix():
    app.logger.info('Request received for getRNNConfussionMatrix')
    return image_to_base64(f'/home/pabloarga/Results/{app.config["SELECTED_RNN_MODEL"]}/confusionMatrix_{app.config["SELECTED_RNN_MODEL"]}.png')

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

@app.route('/api/predict/sequences', methods=['POST'])
def predictSequences():
    app.logger.info('Request received for predict sequences')
    if 'video' not in request.files:
        return jsonify({'error': 'No video uploaded'}), 400
    
    video_file = request.files['video']
    video_name = secure_filename(video_file.filename)
    video_path = os.path.join(app.config['VIDEO_UPLOAD_FOLDER'], video_name)
    video_file.save(video_path)

    # Process the video
    videoFrames, processedSequences = faceExtractor.process_video_to_predict(video_path, sequenceLength = app.config['RNN_MODEL_SEQUENCE_LENGTH'])    
    predictions = modelSequences.predict(np.stack(processedSequences, axis=0))
    mean = np.mean(predictions)
    var = np.var(predictions)
    maxVal = np.max(predictions)
    minVal = np.min(predictions)
    range_ = maxVal - minVal

    resultPredictions = []
    for value in predictions:
        for i in range(app.config['RNN_MODEL_SEQUENCE_LENGTH']):
            resultPredictions.append(float(value[0]))  
        
    # Save the images and get their paths
    video_frame_files = save_sequences(videoFrames, video_name)
    processed_frame_files = save_sequences(processedSequences, f'{video_name}_processed')


    return jsonify({
        'predictions': {
            'id': video_name,
            'data': [{'x': index, 'y': value} for index, value in enumerate(resultPredictions)]
        },
        'mean': float(mean),
        'var':float(var),
        'max':float(maxVal),
        'min':float(minVal),
        'range':float(range_),
        'nSequences': len(processedSequences),
        'sequenceSize': app.config['RNN_MODEL_SEQUENCE_LENGTH'],
        'videoFrames': video_frame_files,
        'processedFrames': processed_frame_files
    }), 200


if __name__ == '__main__':
    app.run(debug=False)
