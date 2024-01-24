import cv2
import os

from sklearn.base import BaseEstimator, TransformerMixin
import concurrent.futures
import pandas as pd

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
        labels = []
        img_count = 0
        for index, row in X.iterrows():
            video_path = row['video']
            label = row['label']
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
                            # Añade la imagen y la etiqueta a los arrays convirtiendolos a float64 e int respectivamente
                            faces.append(face_img)
                            labels.append(int(label))
                            if self.output_dir:
                                videoName = video_path.split('\\')[-1].split('/')[-1].split('.')[0]
                                #crea una carpeta para el video en cuestión
                                if not os.path.exists(os.path.join(self.output_dir, videoName)):
                                    os.makedirs(os.path.join(self.output_dir, videoName))
                                img_count += 1
                                cv2.imwrite(os.path.join(os.path.join(self.output_dir,videoName), f'{videoName}_face_{img_count}.jpg'), face_img)
                else:
                    break
            cap.release()
        df = pd.DataFrame({'face': faces, 'label': labels})
        return df
    



class FaceExtractorMultithread(BaseEstimator, TransformerMixin):
    def __init__(self, n=10, output_dir=None, max_workers=None):
        self.n = n
        self.output_dir = output_dir
        self.max_workers = max_workers
        if self.output_dir and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def fit(self, X, y=None):
        return self

    def process_video(self, row):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        #row es una tupla de [index, row], no nos interesa el index
        row = row[1]
        faces = []
        labels = []
        img_count = 0
        video_path = row['video']
        label = row['label']
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame_count += 1
                if frame_count % self.n == 0:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    detected_faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                    for (x, y, w, h) in detected_faces:
                        face_img = cv2.resize(frame[y:y+h, x:x+w], (64, 64))  # Redimensiona la imagen a 64x64
                        # Añade la imagen y la etiqueta a los arrays
                        faces.append(face_img)
                        labels.append(label)
                        if self.output_dir:
                            videoName = video_path.split('\\')[-1].split('/')[-1].split('.')[0]
                            #crea una carpeta para el video en cuestión
                            if not os.path.exists(os.path.join(self.output_dir, videoName)):
                                os.makedirs(os.path.join(self.output_dir, videoName))
                            img_count += 1
                            cv2.imwrite(os.path.join(os.path.join(self.output_dir,videoName), f'{videoName}_face_{img_count}.jpg'), face_img)
            else:
                break
        cap.release()
        return faces, labels

    def transform(self, X, y=None):
        faces = []
        labels = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Start the futures and store them in a dictionary
            futures = {executor.submit(self.process_video, row): row for _, row in X.iterrows()}
            for future in concurrent.futures.as_completed(futures):
                result_faces, result_labels = future.result()
                faces.extend(result_faces)
                labels.extend(result_labels)
        df = pd.DataFrame({'face': faces, 'label': labels})
        return df
