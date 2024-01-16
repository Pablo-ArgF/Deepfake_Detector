import cv2
import os
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class FaceExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, n=10, output_dir=None):
        self.n = n
        self.output_dir = output_dir
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if self.output_dir and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        faces = []
        img_count = 0
        for video_path in X:
            cap = cv2.VideoCapture(video_path)
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    frame_count += 1
                    if frame_count % self.n == 0:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        detected_faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                        for (x, y, w, h) in detected_faces:
                            face_img = cv2.resize(frame[y:y+h, x:x+w], (64, 64))  # Redimensiona la imagen a 64x64
                            faces.append(face_img)
                            if self.output_dir:
                                img_count += 1
                                cv2.imwrite(os.path.join(self.output_dir, f'face_{img_count}.jpg'), face_img)
                else:
                    break
            cap.release()
        return np.array(faces)


extractor = FaceExtractor(output_dir='imgs')
faces = extractor.fit_transform(['Datasets\CelebDB\Celeb-DF-v2\Celeb-real\id0_0001.mp4'])
