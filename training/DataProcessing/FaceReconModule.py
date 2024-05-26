import cv2
import os
import shutil
from PIL import Image
import math
import random
from sklearn.base import BaseEstimator, TransformerMixin
import concurrent.futures
import pandas as pd
import numpy as np
from scipy import ndimage

class FaceExtractorMultithread(BaseEstimator, TransformerMixin):
    def __init__(self, percentageExtractionFake=0.5, percentageExtractionReal=0.5, max_workers=None):
        self.percentageExtractionFake = percentageExtractionFake # Porcentaje de frames a extraer de los videos fake
        self.percentageExtractionReal = percentageExtractionReal # Porcentaje de frames a extraer de los videos reales
        self.max_workers = max_workers

    """
    Recibe el frame completo, la imagen facial en escala de grises del frame, las coordenadas del rectángulo en el que se encuentra la cara detectada y el clasificador de caras
    Devuelve la cara alineada en el rectángulo del frame con un tamaño de 200x200 píxeles
    """
    def align_frame(self, frame, gray, x, y, w, h, face_cascade): 
        alignedImage,angle = self.align_face(frame, gray, x,y,w,h)
        if angle is None: # Si no se han detectado los ojos, se añade la cara sin alinear
            return frame[y:y+h, x:x+w]

        alignedGray = cv2.cvtColor(alignedImage, cv2.COLOR_BGR2GRAY)
        detected_face = face_cascade.detectMultiScale(alignedGray, 1.255, 4)
        if len(detected_face) == 0: # Si no se ha detectado la cara en la imagen alineada, se añade la cara sin alinear
            return frame[y:y+h, x:x+w]
        xalign = detected_face[0][0]
        yalign = detected_face[0][1]
        walign = detected_face[0][2]
        halign = detected_face[0][3]

        rotatedFace = alignedImage[yalign:yalign+halign, xalign:xalign+walign]
        return cv2.resize(rotatedFace, (200, 200))
        

    def process_video(self, video_path, label):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = []
        labels = []
        cap = cv2.VideoCapture(video_path)

        while cap.isOpened():
            # Leemos un frame del video
            ret, frame = cap.read()
            if ret:
                # Generación aleatoria de número para decidir si el frame se extrae o no
                if (label == 1 and random.random() < self.percentageExtractionFake) or (label == 0 and random.random() < self.percentageExtractionReal): 
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    detected_faces = face_cascade.detectMultiScale(gray, 1.255, 3, minSize=(50, 50))
                    for (x, y, w, h) in detected_faces:
                        alignedFaceImage = self.align_frame(frame, gray[y:y+h, x:x+w],x,y,w,h, face_cascade)
                        faces.append(alignedFaceImage)
                        labels.append(label)                 
            else:
                break
        cap.release()
        return faces, labels
    
    def process_image(self,image_path,label):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = []
        labels = []
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detected_faces = face_cascade.detectMultiScale(gray, 1.305, 7)
        for (x, y, w, h) in detected_faces:
            alignedFaceImage = self.align_frame(img, gray[y:y+h, x:x+w],x,y,w,h, face_cascade)
            faces.append(alignedFaceImage)
            labels.append(label)
        
        return faces,labels



    def process_video_to_predict(self, video_path):
        # Initialize face cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        processed_faces = []  # Processed face frames
        original_frames = []  # Non-cut frames
        cap = cv2.VideoCapture(video_path)
        
        while cap.isOpened():
            # Read a frame from the segment
            ret, frame = cap.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                detected_faces = face_cascade.detectMultiScale(gray, 1.305, 7)
                if len(detected_faces) > 0:
                    x, y, w, h = detected_faces[0]
                    alignedFaceImage = self.align_frame(frame, gray[y:y+h, x:x+w],x,y,w,h, face_cascade)
                    faces.append(alignedFaceImage)
                    original_frames.append(frame)
            else:
                break
        cap.release()

        return np.array(original_frames), np.array(processed_faces)

    def transform(self, videos, videoLabels):
        faces = []
        labels = []
        num_videos = len(videos)
        current = 1
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Start the futures and store them in a dictionary
            futures = {executor.submit(self.process_video, videos[i],videoLabels[i]): i for i in range(num_videos)}
            for future in concurrent.futures.as_completed(futures):
                print(f'Processing video {current}/{num_videos} ----> {math.floor(current/num_videos*100)}%')
                result = future.result()
                faces.extend(result[0])
                labels.extend(result[1])
                current += 1
        return faces,labels
    

    """ 
    Recibe la imagen completa y el rectangulo en el que se encuentra la cara detectada, devuelve el recorte con la cara
    alineada de tal forma que los ojos estén en la misma línea horizontal y la boca en la misma línea vertical
    Si no es capaz de detectar los ojos o de encontrar la cara una vez alineada la imagen devuelve la imagen sin alinear
    """
    def align_face(self, faceImg, greyImg , x,y,w,h):
        eye_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
        eyes = eye_detector.detectMultiScale(greyImg, scaleFactor=1.1203, minNeighbors=3)

        if len(eyes) != 2:
            return faceImg, None
        eye_1 = eyes[0]
        eye_2 = eyes[1]

        if eye_1[0] < eye_2[0]:
            left_eye = eye_1
            right_eye = eye_2
        else:
            left_eye = eye_2
            right_eye = eye_1

        left_eye_center = (int(left_eye[0] + (left_eye[2] / 2)), int(left_eye[1] + (left_eye[3] / 2)))
        right_eye_center = (int(right_eye[0] + (right_eye[2]/2)), int(right_eye[1] + (right_eye[3]/2)))

        # Calculate the angle to rotate the image so that the eyes are horizontal
        dy = right_eye_center[1] - left_eye_center[1]
        dx = right_eye_center[0] - left_eye_center[0]
        angle = np.degrees(np.arctan2(dy, dx))

        # Get the center of the face for the rotation
        center = (x + w / 2, y + h / 2)

        # Calculate the rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Apply the rotation to the image
        rotated = cv2.warpAffine(faceImg, M, (faceImg.shape[1], faceImg.shape[0]))
        return rotated, angle 

"""
TESTS FOR ALIGMENT

# Test the process_video_to_predict method
# https://sefiks.com/2020/02/23/face-alignment-for-face-recognition-in-python-within-opencv/ 
video_path = "C:\\Users\\pablo\\Desktop\\TFG (1)\\TFG (5).mp4"
face_extractor = FaceExtractorMultithread()
faces, _ = face_extractor.process_video(video_path , 1)
#store the images in the folder C:\Users\pablo\Desktop\TFG (1)\test
for i in range(len(faces)):
    cv2.imwrite(f'C:\\Users\\pablo\\Desktop\\TFG (1)\\test\\{i}.jpg',faces[i])

# Test the process_image method
image_path = "C:\\Users\\pablo\\Desktop\\TFG (1)\\0.png"
faces, _ = face_extractor.process_image(image_path , 1)
#store the images in the folder C:\Users\\pablo\\Desktop\\TFG (1)\test
for i in range(len(faces)):
    cv2.imwrite(f'C:\\Users\\pablo\\Desktop\\TFG (1)\\test_image\\{i}.jpg',faces[i])
"""       