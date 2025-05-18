from datetime import datetime
import time
import os
from flask import Flask, jsonify, request, url_for, send_from_directory
from apscheduler.schedulers.background import BackgroundScheduler #Scheduler to delete the added user images 
from tensorflow.keras.models import load_model
import tensorflow as tf
from flask_cors import CORS
import numpy as np
from PIL import Image
import base64
import shutil
from logging.handlers import RotatingFileHandler
from werkzeug.utils import secure_filename
import uuid
import cv2
from io import BytesIO
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
app.config['UPLOAD_FOLDER_REF'] = './results' #REference for the frontend src
app.config['SELECTED_CNN_MODEL'] = '2024-06-20 08.29.00' 
app.config['SELECTED_RNN_MODEL'] = '2024-06-26 16.22.50'
app.config['RNN_MODEL_SEQUENCE_LENGTH'] = 20
app.config['STATIC_IMAGE_FOLDER'] = '/app/static/images/generated'
app.config['LAST_CONV_LAYER_NAME'] = 'concat_2'
EXPIRY_TIME = 3 * 60 * 60  # 3 hours in seconds

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
    with Image.open(image_path) as img:
        img = img.convert('RGB')  # Convert to RGB to avoid transparency issues
        buffered = BytesIO()
        img.save(buffered, format="PNG")  # Save in a consistent format (PNG)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
        
def generate_gradcam_images(model, frames, unique_id):
    gradcam_images = []

    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(app.config['LAST_CONV_LAYER_NAME']).output, model.output]
    )

    for img in frames:
        img_input = np.expand_dims(img, axis=0)

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_input)
            loss = predictions[0]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]

        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1).numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap) + 1e-6

        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

        gradcam_images.append(overlay)
        
    return gradcam_images
    
def generate_heatmap(frames, alpha=0.3, beta=0.7, intensity_multiplier=2.5):
    """
    Genera mapas de calor con diferencias más acentuadas y menor visibilidad de la imagen original.
    
    Args:
        frames (list of np.ndarray): Lista de fotogramas del video en formato numpy array.
        alpha (float): Peso del fotograma original en la superposición (menor = menos visible).
        beta (float): Peso del mapa de calor en la superposición (mayor = más visible).
        intensity_multiplier (float): Multiplicador para intensificar los valores del mapa de calor.
        
    Returns:
        list of np.ndarray: Lista de mapas de calor superpuestos.
    """
    heatmaps = [frames[0]] # El primer fotograma no tiene mapa de calor 
    for i in range(1, len(frames)):
        # Calcular la diferencia absoluta entre fotogramas consecutivos
        diff = cv2.absdiff(frames[i], frames[i - 1])
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        # Intensificar las diferencias
        intensified_diff = cv2.multiply(gray_diff, np.array([intensity_multiplier]))
        intensified_diff = np.clip(intensified_diff, 0, 255).astype(np.uint8)
        
        # Aplicar colormap para crear el mapa de calor
        heatmap = cv2.applyColorMap(intensified_diff, cv2.COLORMAP_JET)
        
        # Superponer el mapa de calor al fotograma original con menor visibilidad de este último
        combined = cv2.addWeighted(frames[i], alpha, heatmap, beta, 0)
        heatmaps.append(combined)
    return heatmaps


def save_images(frames, video_name, unique_id=str(uuid.uuid4()), discard_indices=[]):
    """
    Save frames to a unique folder and return a base URL for requesting the images.
    """
    urls = []
    # Generate a random UUID for the folder
    base_folder = os.path.join(app.config['STATIC_IMAGE_FOLDER'], unique_id)
    os.makedirs(base_folder, exist_ok=True)

    frame_counter = 0
    for i, frame in enumerate(frames):
        # If frame discarted, use next number as counter
        if i in discard_indices:
            frame_counter += 1
        # Ensure the frame is a numpy array
        if isinstance(frame, np.ndarray):
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            image = image.convert('RGB')
        
            # Define the filename and path
            filename = f"{video_name}_frame_{frame_counter}.jpg"
            file_path = os.path.join(base_folder, filename)
            
            # Save the image
            image.save(file_path)  # Saves the image as a .jpg file

            # Construct URL using url_for to serve it from the static folder
            urls.append(url_for('get_image', folder=unique_id, filename=filename, _external=True))
            frame_counter += 1
            
        else:
            app.logger.warning(f"Frame {i} is not a valid numpy array.")
    
    # Return the base URL where images can be accessed
    return urls


@app.route('/api/images/<folder>/<filename>')
def get_image(folder, filename):
    """
    Serve an image file directly from the unique folder.
    """
    base_folder = os.path.join(app.config['STATIC_IMAGE_FOLDER'], folder)
    return send_from_directory(base_folder, filename)

def save_sequences(sequences, video_name):
    """
    Save images from sequences to a unique upload directory.
    Returns a list of file paths or URLs for each saved image.
    """
    image_files = []
    
    # Generate a unique UUID for this batch
    unique_id = str(uuid.uuid4())
    base_folder = os.path.join(app.config['STATIC_IMAGE_FOLDER'], unique_id)
    os.makedirs(base_folder, exist_ok=True)

    for seqId, sequence in enumerate(sequences):
        for i, frame in enumerate(sequence):
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            image = image.convert('RGB')
                    
            # Construct the file path
            file_name = f'{video_name}_sequence_{seqId}_frame_{i}.png'
            file_path = os.path.join(base_folder, file_name)
            
            # Save the frame as an image
            image.save(file_path)

            # Append the reference path or URL for later access
            image_files.append(url_for('get_image', folder=unique_id, filename=file_name, _external=True))

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
    # Generamos un id único para el video en el que se va a guardar todas las imagenes
    unique_id = str(uuid.uuid4())
    video_frame_urls = save_images(videoFrames, 'nonProcessed', unique_id)
    processed_frame_urls = save_images(processedFrames, 'processed', unique_id)

    # Schedule cleanup after 3 hours
    schedule_cleanup(video_name, uploadTime = time.time())

    # Generar mapas de calor
    heatmaps = generate_heatmap(videoFrames)
    heatmaps_face = generate_heatmap(processedFrames)
    # Guardar los mapas de calor y obtener las URLs
    heatmap_urls = save_images(heatmaps, f'heatmap',unique_id)
    heatmap_face_urls = save_images(heatmaps_face, f'heatmap_face',unique_id)

    gradcam_images = generate_gradcam_images(model, processedFrames, unique_id)
    gradcam_urls = save_images(gradcam_images,'gradcam',unique_id)

    return jsonify({
        'uuid': unique_id,
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
        'processedFrames': processed_frame_urls,
        'heatmaps': heatmap_urls ,
        'heatmaps_face':heatmap_face_urls,
        'gradcam_explanations': gradcam_urls

    }), 200

@app.route('/api/recalculate/heatmaps/<uuid>', methods=['POST'])
def recalculateHeatmaps(uuid):
    app.logger.info('Request received for recalculateHeatmaps')
    # Get the video folder
    video_folder = os.path.join(app.config['STATIC_IMAGE_FOLDER'], uuid)
    
    # Check if the folder exists
    if not os.path.exists(video_folder):
        return 'No se encontraron imágenes para el video', 404

    # Get indices to discard from request
    discard_indices = request.json.get('discard_indices', [])

    # two lists, one containing processed frames and the other one the nonProcessed ones
    processed_frames = []
    non_processed_frames = []
    
    # Find all images in folder that start with 'processed' or 'nonProcessed'
    for filename in os.listdir(video_folder):
        frame_index = int(filename.split('_')[-1].split('.')[0])
        if frame_index in discard_indices:
            continue
        if filename.startswith('processed'):
            processed_frames.append(filename)
        elif filename.startswith('nonProcessed'):
            non_processed_frames.append(filename)
    
    # Sort lists by index in name (frame number)
    processed_frames.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    non_processed_frames.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

    # Load the images
    processed_frames = [cv2.imread(os.path.join(video_folder, frame)) for frame in processed_frames]
    non_processed_frames = [cv2.imread(os.path.join(video_folder, frame)) for frame in non_processed_frames]
    
    # Generate heatmaps
    heatmaps = generate_heatmap(non_processed_frames)
    heatmaps_face = generate_heatmap(processed_frames)

    # Guardar los mapas de calor y obtener las URLs
    heatmap_urls = save_images(heatmaps, f'heatmap', uuid, discard_indices)
    heatmap_face_urls = save_images(heatmaps_face, f'heatmap_face', uuid,discard_indices)

    return jsonify({
        'heatmaps': heatmap_urls,
        'heatmaps_face': heatmap_face_urls
    }), 200


    

@app.route('/api/predict/sequences', methods=['POST'])
def predictSequences():
    app.logger.info('Request received for predict sequences')
    
    # Validate video upload
    if 'video' not in request.files:
        return 'No video file uploaded', 400

    if request.files['video'].filename == '':
        return 'The video file is empty', 400

    if not request.files['video'].filename.lower().endswith('.mp4'):
        return 'The file must be a video in MP4 format', 400
    
    # Save the video file
    video_file = request.files['video']
    video_name = secure_filename(video_file.filename)
    video_path = os.path.join(app.config['VIDEO_UPLOAD_FOLDER'], video_name)
    video_file.save(video_path)

    # Process the video to get frames and sequences
    videoFrames, processedSequences = faceExtractor.process_video_to_predict(
        video_path, 
        sequenceLength=app.config['RNN_MODEL_SEQUENCE_LENGTH']
    )

    # Get predictions from the model
    predictions = modelSequences.predict(np.stack(processedSequences, axis=0))

    # Convert all NumPy types to Python-native types
    predictions = [float(value[0]) for value in predictions]
    mean = float(np.mean(predictions))
    var = float(np.var(predictions))
    maxVal = float(np.max(predictions))
    minVal = float(np.min(predictions))
    range_ = float(maxVal - minVal)

    # Expand predictions for each frame in a sequence
    resultPredictions = []
    for value in predictions:
        resultPredictions.extend([value] * app.config['RNN_MODEL_SEQUENCE_LENGTH'])
    
    # Save video frames and get their URLs
    video_frame_urls = save_sequences(videoFrames, video_name)

    # Save processed sequences and get URLs
    processed_frame_urls = save_sequences(processedSequences, f'{video_name}_processed')

    # Schedule cleanup after 3 hours
    schedule_cleanup(video_name, uploadTime=time.time())

    # Generar mapas de calor para secuencias
    heatmaps = generate_heatmap([frame for seq in videoFrames for frame in seq])
    heatmaps_face = generate_heatmap([frame for seq in processedSequences for frame in seq])

    # Guardar los mapas de calor y obtener las URLs
    heatmap_urls = save_images(heatmaps, f'{video_name}_heatmap')
    heatmap_face_urls = save_images(heatmaps_face, f'{video_name}_heatmap_face')

    # Construct response
    response = {
        'predictions': {
            'id': video_name,
            'data': [{'x': index, 'y': value} for index, value in enumerate(resultPredictions)]
        },
        'mean': mean,
        'var': var,
        'max': maxVal,
        'min': minVal,
        'range': range_,
        'nFrames': len(resultPredictions),
        'nSequences': len(processedSequences),
        'sequenceSize': app.config['RNN_MODEL_SEQUENCE_LENGTH'],
        'videoFrames': video_frame_urls,
        'processedFrames': processed_frame_urls,
        'heatmaps': heatmap_urls  ,
        'heatmaps_face': heatmap_face_urls
    }

    app.logger.info(f"Successfully processed video {video_name} and returning predictions.")
    return jsonify(response), 200

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)

