import cv2
import os
import shutil
import math
import random
from sklearn.base import BaseEstimator, TransformerMixin
import concurrent.futures
import pandas as pd
import numpy as np

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
        #Guardamos el numero de videos
        num_videos = len(X)
        for index, row in X.iterrows():
            print(f'Processing video {index}/{num_videos} ----> {math.floor(index/num_videos*100)}%')
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
                            #normalizamos la imagen
                            face_img = face_img / 255.0
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
    
"""
TAKES TOO LONG #TODO documentar esto
"""
class FaceExtractorMultithread_DeepLearning(BaseEstimator, TransformerMixin):
    def __init__(self, n=10, output_dir=None, max_workers=None):
        self.n = n
        self.output_dir = output_dir
        self.max_workers = max_workers
        if self.output_dir and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def fit(self, X, y=None):
        return self

    def process_video(self, video_path, label):
        # Load the pre-trained deep learning face detector
        prototxt_path = "FaceReconModels\deploy.prototxt.txt"  # Path to the network definition
        model_path = "FaceReconModels\\res10_300x300_ssd_iter_140000.caffemodel"  # Path to the pre-trained model
        net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

        faces = []
        labels = []
        img_count = 0
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame_count += 1
                if frame_count % self.n == 0:
                    # Detect faces using the deep learning model
                    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
                    net.setInput(blob)
                    detections = net.forward()

                    for i in range(detections.shape[2]):
                        confidence = detections[0, 0, i, 2]
                        if confidence > 0.5:  # Confidence threshold
                            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                            (x, y, w, h) = box.astype(int)
                            face_img = cv2.resize(frame[y:y+h, x:x+w], (200, 200))
                            faces.append(face_img)
                            labels.append(label)

                            if self.output_dir:
                                video_name = os.path.splitext(os.path.basename(video_path))[0]
                                if not os.path.exists(os.path.join(self.output_dir, video_name)):
                                    os.makedirs(os.path.join(self.output_dir, video_name))
                                img_count += 1
                                cv2.imwrite(os.path.join(self.output_dir, video_name, f"face_{img_count}.jpg"), face_img)

        cap.release()
        return faces, labels
    
    def transform(self, X, y=None):
        faces = []
        labels = []
        num_videos = len(X)
        current = 1
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Start the futures and store them in a dictionary
            futures = {executor.submit(self.process_video, row['video'],row['label']): row for _, row in X.iterrows()}
            for future in concurrent.futures.as_completed(futures):
                print(f'Processing video {current}/{num_videos} ----> {math.floor(current/num_videos*100)}%')
                result = future.result()
                faces.extend(result[0])
                labels.extend(result[1])
                current += 1
        # TODO echarle un ojo a ver si las threads estan cerrandose executor.shutdown(wait=False)  # Añadido para asegurar que todos los hilos finalicen
        df = pd.DataFrame({'face': faces, 'label': labels})
        return df


class FaceExtractorMultithread(BaseEstimator, TransformerMixin):
    def __init__(self, percentageExtractionFake=0.5, percentageExtractionReal=0.5, max_workers=None):
        self.percentageExtractionFake = percentageExtractionFake # Porcentaje de frames a extraer de los videos fake
        self.percentageExtractionReal = percentageExtractionReal # Porcentaje de frames a extraer de los videos reales
        self.max_workers = max_workers
        

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
                    detected_faces = face_cascade.detectMultiScale(gray, 1.305, 7)
                    for (x, y, w, h) in detected_faces:
                        face_img = cv2.resize(frame[y:y+h, x:x+w], (200, 200))
                        faces.append(face_img)
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
            face_img = cv2.resize(img[y:y+h, x:x+w], (200, 200))
            faces.append(face_img)
            labels.append(label)
        
        return faces,labels




    def process_video_to_predict(video_path):
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
                    # Get the face that is most centered
                    x, y, w, h = detected_faces[0]
                    face_img = cv2.resize(frame[y:y+h, x:x+w], (200, 200))
                    processed_faces.append(face_img)
                original_frames.append(frame)
            else:
                break
        cap.release()

        return original_frames, processed_faces


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
#probamos la version multithread
paths = ['Datasets\CelebDB\Celeb-DF-v2\Celeb-real\id0_0000.mp4'
                             ,'Datasets\CelebDB\Celeb-DF-v2\Celeb-real\id0_0000.mp4'
                             ,'Datasets\CelebDB\Celeb-DF-v2\Celeb-real\id0_0000.mp4'
                             ,'Datasets\CelebDB\Celeb-DF-v2\Celeb-real\id0_0000.mp4'
                             ,'Datasets\CelebDB\Celeb-DF-v2\Celeb-real\id0_0000.mp4'
                             ,'Datasets\CelebDB\Celeb-DF-v2\Celeb-real\id0_0000.mp4'
                             ,'Datasets\CelebDB\Celeb-DF-v2\Celeb-real\id0_0000.mp4'
                             ,'Datasets\CelebDB\Celeb-DF-v2\Celeb-real\id0_0000.mp4'
                             ,'Datasets\CelebDB\Celeb-DF-v2\Celeb-real\id0_0000.mp4']
labels =  [1,1,1,1,1,1,1,1,1]
face_extractor = FaceExtractorMultithread()
faces,labels= face_extractor.transform(paths,labels)
print(len(faces))
print(len(labels))
#numero de 0 y 1s en labels
print(sum(labels))
"""

"""
class FeaturesExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, n=10,output_dir=None):
        self.n = n
        self.output_dir = output_dir
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.mouth_cascade = cv2.CascadeClassifier(filename='.\haarcascade\haarcascade_mcs_mouth.xml')
        self.nose_cascade = cv2.CascadeClassifier(filename='.\haarcascade\haarcascade_mcs_nose.xml')
        #creamos el directorio de salida si no existe
        if self.output_dir and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)


    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        img_count = 0
        features = []
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
                        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                        for (x, y, w, h) in faces:
                            roi_gray = gray[y:y+h, x:x+w]
                            roi_color = frame[y:y+h, x:x+w]
                            eyes = self.eye_cascade.detectMultiScale(roi_gray)
                            mouth = self.mouth_cascade.detectMultiScale(roi_gray)
                            nose = self.nose_cascade.detectMultiScale(roi_gray)
                            if isinstance(eyes, np.ndarray):
                                eyes = eyes.tolist()
                            if isinstance(mouth, np.ndarray):
                                mouth = mouth.tolist()
                            if isinstance(nose, np.ndarray):
                                nose = nose.tolist()
                            features.append([eyes, mouth, nose, label])
                            if self.output_dir:
                                videoName = video_path.split('\\')[-1].split('/')[-1].split('.')[0]
                                #crea una carpeta para el video en cuestión
                                if not os.path.exists(os.path.join(self.output_dir, videoName)):
                                    os.makedirs(os.path.join(self.output_dir, videoName))
                                    #creamos una carpeta por cada feature
                                    os.makedirs(os.path.join(os.path.join(self.output_dir, videoName),'eyes'))
                                    os.makedirs(os.path.join(os.path.join(self.output_dir, videoName),'mouth'))
                                    os.makedirs(os.path.join(os.path.join(self.output_dir, videoName),'nose'))
                                img_count += 1
                                #guardamos las imagenes de los features
                                #ojos
                                for i,eye in enumerate(eyes):
                                    cv2.imwrite(os.path.join(os.path.join(os.path.join(self.output_dir,videoName),'left_eye'), f'{videoName}_face_{img_count}_eye_{i}.jpg'), roi_color[eye[1]:eye[1]+eye[3], eye[0]:eye[0]+eye[2]])
                                #boca
                                for i,m in enumerate(mouth):
                                    cv2.imwrite(os.path.join(os.path.join(os.path.join(self.output_dir,videoName),'mouth'), f'{videoName}_face_{img_count}_mouth_{i}.jpg'), roi_color[m[1]:m[1]+m[3], m[0]:m[0]+m[2]])
                                #nariz
                                for i,n in enumerate(nose):
                                    cv2.imwrite(os.path.join(os.path.join(os.path.join(self.output_dir,videoName),'nose'), f'{videoName}_face_{img_count}_nose_{i}.jpg'), roi_color[n[1]:n[1]+n[3], n[0]:n[0]+n[2]])
                                
                else:
                    break
            cap.release()
        return pd.DataFrame(features, columns=['eyes', 'mouth', 'nose', 'label'])
    

#probamos la version con features
df0 = pd.DataFrame({'video': ['Datasets\CelebDB\Celeb-DF-v2\Celeb-real\id0_0000.mp4'
                             ,'Datasets\CelebDB\Celeb-DF-v2\Celeb-real\id0_0001.mp4'
                             ,'Datasets\CelebDB\Celeb-DF-v2\Celeb-real\id0_0002.mp4'
                             ,'Datasets\CelebDB\Celeb-DF-v2\Celeb-real\id0_0003.mp4'
                             ,'Datasets\CelebDB\Celeb-DF-v2\Celeb-real\id0_0004.mp4'
                             ,'Datasets\CelebDB\Celeb-DF-v2\Celeb-real\id0_0005.mp4'
                             ,'Datasets\CelebDB\Celeb-DF-v2\Celeb-real\id0_0006.mp4'
                             ,'Datasets\CelebDB\Celeb-DF-v2\Celeb-real\id0_0007.mp4'
                             ,'Datasets\CelebDB\Celeb-DF-v2\Celeb-real\id0_0008.mp4'], 'label': [1,1,1,1,1,1,1,1,1]})
face_extractor = FeaturesExtractor(n=100,output_dir='imgs')
df= face_extractor.transform(df0)
print(df.head())
print(df.info())
print(df.describe())
print('------------------------------------------------------')
face_extractor = FaceExtractor(n=100)
df= face_extractor.transform(df0)
print(df.head())
print(df.info())
print(df.describe())

"""