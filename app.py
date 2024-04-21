# app.py
from flask import Flask, request, jsonify
import tensorflow as tf
import cv2
import numpy as np
#Base 64 frame transformation
import base64
from PIL import Image
from io import BytesIO
#My modules
from training.DataProcessing.FaceReconModule import FaceExtractorMultithread

import sys
sys.path.append("..")

app = Flask(__name__)

# Load your TensorFlow model
model = tf.keras.models.load_model('/home/pabloarga/Results/2024-04-02 16.26.25/model2024-04-02 16.26.25.keras',safe_mode=False,compile=False)
faceExtractor = FaceExtractorMultithread() 
video_path = '/tmp/video.mp4'

def encondeBase64(images):
    """
    Converts an array of images (in vector form) to base64-encoded images.

    Args:
        images (List[np.ndarray]): List of images (numpy arrays).

    Returns:
        List[str]: List of base64-encoded images.
    """
    base64_images = []
    for img in images:
        pil_img = Image.fromarray(img)
        buff = BytesIO()
        pil_img.save(buff, format="JPEG")
        img_str = base64.b64encode(buff.getvalue()).decode("utf-8")
        base64_images.append(img_str)
    return base64_images

@app.route('/predict', methods=['POST'])
def predict():
    if 'video' not in request.files:
        return jsonify({'error': 'No video uploaded'}), 400
    
    video_file = request.files['video']
    video_file.save(video_path)
    # Store video name
    video_name = video_file.filename

    #Store the video frames without processing
    videoFrames = []
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            videoFrames.append(frame)
        else:
            break
    
    # Process the video
    processedFrames = faceExtractor.process_video_to_predict(video_path)


    # Convert frames to base64
    videoFrames64 = encondeBase64(videoFrames)
    processedFrames64 = encondeBase64(processedFrames)

    # Make predictions
    predictions = model.predict(np.stack(processedFrames, axis=0))

    # Convert predictions to regular Python list and convert numpy.float32 to Python float
    predictions = [float(value) for value in predictions]

    mean = np.mean(predictions)

    return jsonify({'predictions': {"id": video_name, "data" : [{"x": index, "y": value} for index,value in enumerate(predictions)] }
                     , 'mean' : mean, "nFrames" : len(predictions), 'videoFrames': videoFrames64 , 'processedFrames': processedFrames64}), 200

if __name__ == '__main__':
    app.run(debug=True)
