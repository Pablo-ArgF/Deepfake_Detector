from keras.models import load_model
import cv2
from training.DataProcessing.FaceReconModule import FaceExtractorMultithread
import numpy as np

#cargamos el modelo del escritorio
modelPath = '/home/pabloarga/Results/2024-03-25 22.15.24/model2024-03-25 22.15.24.keras'
videoTest = "/home/pabloarga/testVideos/realVideo2.mp4"

#Cargamos el modelo
model = load_model(modelPath,safe_mode=False,compile=False)
#obtenemos el número de frames del video
#cap = cv2.VideoCapture(videoTest)
#nFramesVideo = cap.get(cv2.CAP_PROP_FRAME_COUNT)
#Procesamos el video frame por frame
faceExtractor = FaceExtractorMultithread(50,output_dir='/home/pabloarga/testVideos/outputDir/') #
frames = faceExtractor.process_video(videoTest,1)[0]

#probamos el modelo manualmente
y_pred = model.predict(np.stack(frames, axis=0))
print(y_pred)
print(np.mean(y_pred))