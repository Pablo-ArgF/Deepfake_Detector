from datetime import datetime
import time
import os
from flask import Flask, jsonify, request, url_for, send_from_directory
from apscheduler.schedulers.background import BackgroundScheduler #Scheduler to delete the added user images 
from tensorflow.keras.models import load_model
import tensorflow as tf
from flask_cors import CORS
import numpy as np

# Enable XLA JIT compilation for maximum performance
tf.config.optimizer.set_jit(True)

from PIL import Image
import base64
import shutil
from logging.handlers import RotatingFileHandler
from werkzeug.utils import secure_filename
import uuid
import cv2
import json
from io import BytesIO
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

# Load environment variables
load_dotenv()

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
EXPIRY_TIME = 5 * 60 * 60  # 5 hours in seconds

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

def predict_in_batches(model, data, batch_size=16):
    """
    Predict in small batches to avoid OOM when converting large lists to tensors.
    """
    preds = []
    for i in range(0, len(data), batch_size):
        chunk = data[i:i + batch_size]
        batch = np.stack(chunk, axis=0)
        batch_preds = model.predict(batch, verbose=0)
        # Handle both single-value (CNN) and multi-value (RNN/Sequences) outputs
        if len(batch_preds.shape) > 1 and batch_preds.shape[1] == 1:
            preds.extend(batch_preds.flatten().tolist())
        else:
            preds.extend(batch_preds.tolist())
    return preds

def get_adaptive_skip(video_path):
    """
    Calculate how many frames to skip based on video length to maintain performance.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    if total_frames > 3000: # > ~1.5 mins at 30fps
        return 2 # Process every 3rd frame
    if total_frames > 1000: # > ~33s at 30fps
        return 1 # Process every 2nd frame
    return 0

def image_to_base64(image_path):
    with Image.open(image_path) as img:
        img = img.convert('RGB')  # Convert to RGB to avoid transparency issues
        buffered = BytesIO()
        img.save(buffered, format="PNG")  # Save in a consistent format (PNG)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
        
def generate_gradcam_images(model, frames, unique_id):
    """
    Vectorized Grad-CAM generation with batching to avoid memory issues.
    """
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(app.config['LAST_CONV_LAYER_NAME']).output, model.output]
    )
    
    gradcam_images = []
    batch_size = 16
    
    for i in range(0, len(frames), batch_size):
        batch_frames = frames[i:i + batch_size]
        img_input = np.stack(batch_frames, axis=0)
        
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_input)
            loss = predictions[:, 0] if len(predictions.shape) > 1 else predictions[0]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(1, 2))
        
        for j in range(len(batch_frames)):
            # Weighted average of channels
            heatmap = tf.reduce_sum(tf.multiply(pooled_grads[j], conv_outputs[j]), axis=-1).numpy()
            heatmap = np.maximum(heatmap, 0)
            heatmap /= (np.max(heatmap) + 1e-6)

            heatmap = cv2.resize(heatmap, (batch_frames[j].shape[1], batch_frames[j].shape[0]))
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(batch_frames[j], 0.6, heatmap, 0.4, 0)
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


def save_frame_to_disk(frame, base_folder, filename):
    """Worker function for parallel saving."""
    if isinstance(frame, np.ndarray):
        file_path = os.path.join(base_folder, filename)
        # Fast write with OpenCV (BGR is already the format)
        cv2.imwrite(file_path, frame)
        return True
    return False

def save_images(frames, video_name, unique_id=str(uuid.uuid4()), discard_indices=[]):
    """
    Save frames to a unique folder using multiple threads for speed.
    """
    urls = []
    base_folder = os.path.join(app.config['STATIC_IMAGE_FOLDER'], unique_id)
    os.makedirs(base_folder, exist_ok=True)

    tasks = []
    frame_counter = 0
    
    with ThreadPoolExecutor(max_workers=min(32, os.cpu_count() + 4)) as executor:
        for i, frame in enumerate(frames):
            if i in discard_indices:
                frame_counter += 1
                continue
                
            filename = f"{video_name}_frame_{frame_counter}.jpg"
            executor.submit(save_frame_to_disk, frame, base_folder, filename)
            
            # Construct URL (optimized: avoid url_for overhead in loop)
            # urls.append(url_for('get_image', folder=unique_id, filename=filename, _external=True))
            # Manual construction is safer for bulk
            urls.append(f"/api/images/{unique_id}/{filename}")
            frame_counter += 1
            
    return urls


@app.route('/api/images/<folder>/<filename>')
def get_image(folder, filename):
    """
    Serve an image file directly from the unique folder.
    """
    base_folder = os.path.join(app.config['STATIC_IMAGE_FOLDER'], folder)
    return send_from_directory(base_folder, filename)

@app.route('/api/video/<uuid>')
def get_video(uuid):
    """
    Serve the video file directly for a given UUID.
    """
    base_folder = os.path.join(app.config['STATIC_IMAGE_FOLDER'], uuid)
    return send_from_directory(base_folder, 'video.mp4')

def save_sequences(sequences, video_name):
    """
    Save images from sequences to a unique upload directory using multiple threads.
    """
    image_files = []
    unique_id = str(uuid.uuid4())
    base_folder = os.path.join(app.config['STATIC_IMAGE_FOLDER'], unique_id)
    os.makedirs(base_folder, exist_ok=True)

    with ThreadPoolExecutor(max_workers=min(32, os.cpu_count() + 4)) as executor:
        for seqId, sequence in enumerate(sequences):
            for i, frame in enumerate(sequence):
                file_name = f'{video_name}_sequence_{seqId}_frame_{i}.png'
                executor.submit(save_frame_to_disk, frame, base_folder, file_name)
                image_files.append(f"/api/images/{unique_id}/{file_name}")

    return image_files, unique_id
    
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

def verify_password():
    password = request.headers.get('X-Upload-Password')
    expected_password = os.getenv('UPLOAD_PASSWORD')
    if not expected_password:
        return True
    return password == expected_password
def run_background_cnn_processing(unique_id, processedFrames, video_path, video_name, analysis_results):
    """
    Heavy lifting done in the background. Loads original frames from disk to avoid RAM OOM.
    """
    base_folder = os.path.join(app.config['STATIC_IMAGE_FOLDER'], unique_id)
    
    # Load original frames from disk for heatmap generation
    # We use processedFrames length because that's how many frames we have successful detections for
    n_frames = len(processedFrames)
    videoFrames = []
    for i in range(n_frames):
        img_path = os.path.join(base_folder, f"nonProcessed_frame_{i}.jpg")
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                videoFrames.append(img)
    
    # Ensure processedFrames is a numpy array if it came as a list
    if isinstance(processedFrames, list):
        processedFrames = np.array(processedFrames)

    with ThreadPoolExecutor(max_workers=4) as executor:
        # 1. Heatmaps
        f_hm = executor.submit(generate_heatmap, videoFrames)
        f_hm_face = executor.submit(generate_heatmap, list(processedFrames))
        
        # 2. Grad-CAM (vectorized batching)
        f_gc = executor.submit(generate_gradcam_images, model, processedFrames, unique_id)

        heatmaps = f_hm.result()
        heatmaps_face = f_hm_face.result()
        gradcam_images = f_gc.result()

        executor.submit(save_images, heatmaps, 'heatmap', unique_id)
        executor.submit(save_images, heatmaps_face, 'heatmap_face', unique_id)
        executor.submit(save_images, gradcam_images, 'gradcam', unique_id)

    results_path = os.path.join(base_folder, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(analysis_results, f)

@app.route('/api/predict', methods=['POST'])
def predict():
    app.logger.info('Request received for predict (Aggressive Optimization)')
    
    if not verify_password():
        return jsonify({'error': 'Incorrect or missing password.'}), 403

    if 'video' not in request.files:
        return 'No se ha subido ningún video', 400
    
    unique_id = str(uuid.uuid4())
    video_file = request.files['video']
    video_name = secure_filename(video_file.filename)
    
    base_folder = os.path.join(app.config['STATIC_IMAGE_FOLDER'], unique_id)
    os.makedirs(base_folder, exist_ok=True)

    # Save video directly to final destination as video.mp4
    video_path = os.path.join(base_folder, "video.mp4")
    video_file.save(video_path)

    # 1. Prediction (WAIT)
    # Adaptive stride for performance
    skip = get_adaptive_skip(video_path)
    app.logger.info(f"Using skip_frames={skip} for video {video_name}")

    _, processedFrames = faceExtractor.process_video_to_predict(
        video_path, unique_id=unique_id, skip_frames=skip
    )    
    
    if len(processedFrames) == 0:
        return jsonify({'error': 'No faces detected in the video.'}), 400

    # Batch prediction to avoid OOM
    predictions = predict_in_batches(model, processedFrames, batch_size=16)
    predictions = [float(value) for value in predictions]
    
    # 2. Prepare Results (INSTANT)
    analysis_results = {
        'uuid': unique_id,
        'type': 'cnn',
        'predictions': {
            'id': video_name,
            'data': [{'x': index, 'y': value} for index, value in enumerate(predictions)]
        },
        'mean': float(np.mean(predictions)),
        'var': float(np.var(predictions)),
        'max': float(np.max(predictions)),
        'min': float(np.min(predictions)),
        'range': float(np.max(predictions) - np.min(predictions)),
        'nFrames': len(predictions),
        'videoFrames': [f"/api/images/{unique_id}/nonProcessed_frame_{i}.jpg" for i in range(len(predictions))],
        'processedFrames': [f"/api/images/{unique_id}/processed_frame_{i}.jpg" for i in range(len(predictions))],
        'heatmaps': [f"/api/images/{unique_id}/heatmap_frame_{i}.jpg" for i in range(len(predictions))],
        'heatmaps_face': [f"/api/images/{unique_id}/heatmap_face_frame_{i}.jpg" for i in range(len(predictions))],
        'gradcam_explanations': [f"/api/images/{unique_id}/gradcam_frame_{i}.jpg" for i in range(len(predictions))]
    }

    results_path = os.path.join(base_folder, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(analysis_results, f)

    # 3. Background heavy worker
    import threading
    threading.Thread(target=run_background_cnn_processing, args=(unique_id, processedFrames, video_path, video_name, analysis_results)).start()

    schedule_cleanup(video_name, uploadTime=time.time())
    return jsonify(analysis_results), 200

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


    

def run_background_rnn_processing(rnn_uuid, processed_sequences, video_path, video_name, response):
    """
    RNN background processing. Metadata update only.
    """
    base_folder = os.path.join(app.config['STATIC_IMAGE_FOLDER'], rnn_uuid)
    
    # Heavy image processing removed for RNN as per user request
    results_path = os.path.join(base_folder, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(response, f)

@app.route('/api/predict/sequences', methods=['POST'])
def predictSequences():
    app.logger.info('Request received for predict sequences (Aggressive Optimization)')

    if not verify_password():
        return jsonify({'error': 'Incorrect or missing password.'}), 403

    # Check if we are continuing an existing analysis or starting a new one
    rnn_uuid = request.form.get('uuid') or request.args.get('uuid')
    
    if rnn_uuid:
        base_folder = os.path.join(app.config['STATIC_IMAGE_FOLDER'], rnn_uuid)
        video_path = os.path.join(base_folder, "video.mp4")
        if not os.path.exists(video_path):
             return jsonify({'error': 'Existing analysis folder or video not found.'}), 404
        # We assume video_name is not strictly needed or can be inferred
        video_name = "reanalyzed_video" 
    else:
        if 'video' not in request.files:
            return 'No video file uploaded', 400
        rnn_uuid = str(uuid.uuid4())
        video_file = request.files['video']
        video_name = secure_filename(video_file.filename)
        base_folder = os.path.join(app.config['STATIC_IMAGE_FOLDER'], rnn_uuid)
        os.makedirs(base_folder, exist_ok=True)
        video_path = os.path.join(base_folder, "video.mp4")
        video_file.save(video_path)

    # 1. Prediction (STREAMING to avoid OOM)
    # For RNN, we use a smaller skip to preserve some temporal context 
    skip = min(get_adaptive_skip(video_path), 1)
    
    predictions = []
    n_sequences = 0
    seq_len = app.config['RNN_MODEL_SEQUENCE_LENGTH']

    # Use the new streaming method
    for sequence in faceExtractor.stream_video_to_predict(
        video_path, 
        unique_id=rnn_uuid, 
        sequenceLength=seq_len,
        skip_frames=skip,
        save_original_frames=True # Save original frames as requested
    ):
        # Predict one sequence at a time
        # Expand dims to (1, seq_len, 200, 200, 3)
        batch = np.expand_dims(sequence, axis=0)
        pred_raw = modelSequences.predict(batch, verbose=0)
        
        # Extract float value
        if isinstance(pred_raw, np.ndarray) and len(pred_raw.shape) > 1:
            val = float(pred_raw[0][0])
        else:
            val = float(pred_raw)
        
        predictions.append(val)
        n_sequences += 1
    
    if n_sequences == 0:
        return jsonify({'error': 'No sequences detected in the video.'}), 400

    resultPredictions = []
    for value in predictions:
        resultPredictions.extend([value] * seq_len)

    # 2. Prepare Result (INSTANT)
    response = {
        'uuid': rnn_uuid,
        'type': 'rnn',
        'predictions': {
            'id': video_name,
            'data': [{'x': index, 'y': value} for index, value in enumerate(resultPredictions)]
        },
        'mean': float(np.mean(predictions)) if predictions else 0,
        'var': float(np.var(predictions)) if predictions else 0,
        'max': float(np.max(predictions)) if predictions else 0,
        'min': float(np.min(predictions)) if predictions else 0,
        'range': float(np.max(predictions) - np.min(predictions)) if predictions else 0,
        'nFrames': len(resultPredictions),
        'nSequences': n_sequences,
        'sequenceSize': seq_len,
        'videoFrames': [
            f"/api/images/{rnn_uuid}/sequence_{i//seq_len}_frame_{i%seq_len}.png" 
            for i in range(len(resultPredictions))
        ],
        'processedFrames': [
            f"/api/images/{rnn_uuid}/processed_sequence_{i//seq_len}_frame_{i%seq_len}.png" 
            for i in range(len(resultPredictions))
        ]
    }

    results_path = os.path.join(base_folder, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(response, f)

    # 3. Fire-and-forget (Background metadata update)
    import threading
    threading.Thread(target=run_background_rnn_processing, args=(rnn_uuid, [], video_path, video_name, response)).start()

    schedule_cleanup(video_name, uploadTime=time.time())
    return jsonify(response), 200

@app.route('/api/results/<uuid>', methods=['GET'])
def get_results(uuid):
    """
    Retrieve saved analysis results for a given UUID.
    """
    results_path = os.path.join(app.config['STATIC_IMAGE_FOLDER'], uuid, 'results.json')
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            data = json.load(f)
        return jsonify(data), 200
    return 'Results not found', 404

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)

