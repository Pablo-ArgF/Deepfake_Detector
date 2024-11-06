from datetime import datetime
import time
import os
from flask import Flask, jsonify, request, url_for
from apscheduler.schedulers.background import BackgroundScheduler #Scheduler to delete the added user images 
from tensorflow.keras.models import load_model
from flask_cors import CORS
import numpy as np
from PIL import Image
import base64
import shutil
from logging.handlers import RotatingFileHandler
from werkzeug.utils import secure_filename
import sys
import cv2
# Add the backend directory to the Python path
from training.DataProcessing.DataProcessor import FaceExtractorMultithread


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
app.config['UPLOAD_FOLDER'] = '/home/pabloarga/Deepfake_Detector/frontend/build/results' #TODO remove
app.config['UPLOAD_FOLDER_REF'] = './results' #REference for the frontend src
app.config['SELECTED_CNN_MODEL'] = '2024-06-20 08.29.00' 
app.config['SELECTED_RNN_MODEL'] = '2024-06-26 16.22.50'
app.config['RNN_MODEL_SEQUENCE_LENGTH'] = 20
app.config['STATIC_IMAGE_FOLDER'] = '/path/to/static/frames'
EXPIRY_TIME = 60 #3 * 60 * 60  # 3 hours in seconds

# Load the cnn model
path = f"/app/models/{app.config['SELECTED_CNN_MODEL']}/model{app.config['SELECTED_CNN_MODEL']}.keras"  
model = load_model(path, safe_mode=False, compile=False)

# Load the rnn model
pathSequences = f"/app/models/{app.config['SELECTED_RNN_MODEL']}/model{app.config['SELECTED_RNN_MODEL']}.keras"  
modelSequences = load_model(pathSequences, safe_mode=False, compile=False)

#---------------------------------------------------
# Job to delete the images on the server after EXPIRY_TIME time
def cleanup_old_files(video_name):
    """Cleanup function to remove video and frame files older than 3 hours."""
    video_folder = os.path.join(app.config['STATIC_IMAGE_FOLDER'], video_name)
    
    if os.path.exists(video_folder):
        shutil.rmtree(video_folder)
        app.logger.info(f"Deleted images for video: {video_name}")
    
    video_file_path = os.path.join(app.config['VIDEO_UPLOAD_FOLDER'], video_name)
    if os.path.exists(video_file_path):
        os.remove(video_file_path)
        app.logger.info(f"Deleted video file: {video_name}")

def schedule_cleanup(video_name, uploadTime):
    """Schedule cleanup for each video after 3 hours."""
    delay = EXPIRY_TIME - (time.time() - uploadTime)
    if delay > 0:
        run_date = datetime.fromtimestamp(time.time() + delay)
        scheduler.add_job(func=cleanup_old_files, args=[video_name], trigger='date', run_date=run_date)
        app.logger.info(f"Scheduled cleanup for video {video_name} in {delay/3600:.2f} hours.")

# Initialize APScheduler
scheduler = BackgroundScheduler()
scheduler.start()
#---------------------------------------------------
 

faceExtractor = FaceExtractorMultithread() 

def image_to_base64(image_path):
    with open(image_path, 'rb') as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def save_images(frames, video_name):
    """
    Save frames to a static folder and return a list of their URLs.
    """
    urls = []
    base_folder = os.path.join(app.config['STATIC_IMAGE_FOLDER'], video_name)
    os.makedirs(base_folder, exist_ok=True)
    
    for i, frame in enumerate(frames):
        # Ensure the frame is a numpy array
        if isinstance(frame, np.ndarray):
            # Convert the numpy array to a PIL image
            image = Image.fromarray(frame)
        
            # Define the filename and path
            filename = f"{video_name}_frame_{i}.jpg"
            file_path = os.path.join(base_folder, filename)
            
            # Save the image
            image.save(file_path)  # Saves the image as a .jpg file

            # Construct URL using url_for to serve it from the static folder
            urls.append(url_for('static', filename=f'frames/{video_name}/{filename}', _external=True))
        else:
            app.logger.warning(f"Frame {i} is not a valid numpy array.")
    
    return urls

def save_sequences(sequences, video_name):
    """
    Guarda las imagenes de las secuencias en el directorio de carga.
    """
    image_files = []
    for seqId, sequence in enumerate(sequences):
        for i, frame in enumerate(sequence):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{video_name}_sequence_{seqId}_frame_{i}.png')
            cv2.imwrite(file_path, frame)
            image_files.append(os.path.join(app.config['UPLOAD_FOLDER_REF'], f'{video_name}_sequence_{seqId}_frame_{i}.png'))
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
    return image_to_base64(f"/app/models/{app.config['SELECTED_CNN_MODEL']}/model_structure.png")

@app.route('/api/model/structure/rnn', methods=['GET'])
def getRNNStructure():
    app.logger.info('Request received for getRNNStructure')
    return image_to_base64(f"/app/models/{app.config['SELECTED_RNN_MODEL']}/model_structure.png")

@app.route('/api/model/graphs/cnn', methods=['GET'])
def getCNNGraphs():
    app.logger.info('Request received for getCNNGraphs')
    return image_to_base64(f"/app/models/{app.config['SELECTED_CNN_MODEL']}/combined_plots.png")

@app.route('/api/model/graphs/rnn', methods=['GET'])
def getRNNGraphs():
    app.logger.info('Request received for getRNNGraphs')
    return image_to_base64(f"/app/models/{app.config['SELECTED_RNN_MODEL']}/combined_plots.png")

@app.route('/api/model/confussion/matrix/cnn', methods=['GET'])
def getCNNConfussionMatrix():
    app.logger.info('Request received for getCNNConfussionMatrix')
    return image_to_base64(f"/app/models/{app.config['SELECTED_CNN_MODEL']}/confusionMatrix_{app.config['SELECTED_CNN_MODEL']}.png")

@app.route('/api/model/confussion/matrix/rnn', methods=['GET'])
def getRNNConfussionMatrix():
    app.logger.info('Request received for getRNNConfussionMatrix')
    return image_to_base64(f"/app/models/{app.config['SELECTED_RNN_MODEL']}/confusionMatrix_{app.config['SELECTED_RNN_MODEL']}.png")

@app.route('/api/predict', methods=['POST'])
def predict():
    app.logger.info('Request received for predict')
    if 'video' not in request.files:
        return 'No se ha subido ningún video', 400
    
    if request.files['video'].filename == '':
        return 'El archivo de video está vacío', 400

    if not request.files['video'].filename.lower().endswith('.mp4'):
        return 'El archivo debe ser un video en formato MP4', 400

    
    
    video_file = request.files['video']
    video_name = secure_filename(video_file.filename)

    #Remove previous frame files
    #remove_all_files(app.config['UPLOAD_FOLDER'])
    
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

    # Save images and get URLs
    video_frame_urls = save_images(videoFrames, video_name)
    processed_frame_urls = save_images(processedFrames, f'{video_name}_processed')

    # Schedule cleanup after 3 hours
    schedule_cleanup(video_name, uploadTime = time.time())

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
        'videoFrames': video_frame_urls,
        'processedFrames': processed_frame_urls
    }), 200

@app.route('/api/predict/sequences', methods=['POST'])
def predictSequences():
    app.logger.info('Request received for predict sequences')
    if 'video' not in request.files:
        return 'No se ha subido ningún video', 400

    if request.files['video'].filename == '':
        return 'El archivo de video está vacío', 400

    if not request.files['video'].filename.lower().endswith('.mp4'):
        return 'El archivo debe ser un video en formato MP4', 400
    
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
        
    # Save images and get URLs
    video_frame_urls = save_images(videoFrames, video_name)
    processed_frame_urls = save_images(processedSequences, f'{video_name}_processed')

    # Schedule cleanup after 3 hours
    schedule_cleanup(video_name, upload_time = time.time())

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
        'videoFrames': video_frame_urls,
        'processedFrames': processed_frame_urls
    }), 200


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)

