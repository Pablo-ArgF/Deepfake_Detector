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
from concurrent.futures import ThreadPoolExecutor

class FaceExtractorMultithread(BaseEstimator, TransformerMixin):
    """
    Clase que se encarga de extraer caras de imágenes y videos utilizando múltiples hilos de ejecución.

    :param percentageExtractionFake: Porcentaje de frames a extraer de los videos fake.
    :type percentageExtractionFake: float
    :param percentageExtractionReal: Porcentaje de frames a extraer de los videos reales.
    :type percentageExtractionReal: float
    :param max_workers: Número máximo de hilos de ejecución.
    :type max_workers: int

    :ivar percentageExtractionFake: Porcentaje de frames a extraer de los videos fake.
    :vartype percentageExtractionFake: float
    :ivar percentageExtractionReal: Porcentaje de frames a extraer de los videos reales.
    :vartype percentageExtractionReal: float
    :ivar max_workers: Número máximo de hilos de ejecución.
    :vartype max_workers: int

    :method align_frame: Alinea una cara en un rectángulo del frame.
    :method process_video: Procesa un video y extrae las caras.
    :method process_image: Procesa una imagen y extrae las caras.
    :method process_frame_chunk: Procesa un fragmento de frames y extrae las caras.
    :method process_video_to_predict: Procesa un video para predecir y extrae las caras.
    :method transform: Transforma los videos y extrae las caras.
    """


    def __init__(self, percentageExtractionFake=0.5, percentageExtractionReal=0.5, max_workers=None):
        """
        Inicia el objeto FaceExtractorMultithread
        Args:
            percentageExtractionFake (float): Porcentaje de frames a extraer de los videos fake.
            percentageExtractionReal (float): Porcentaje de frames a extraer de los videos reales.
            max_workers (int): Número máximo de hilos de ejecución.
        """
        self.percentageExtractionFake = percentageExtractionFake
        self.percentageExtractionReal = percentageExtractionReal
        self.max_workers = max_workers

    def align_frame(self, frame, gray, x, y, w, h, face_cascade):
        """
        Alinea una cara en un rectángulo del frame con un tamaño de 200x200 píxeles.

        Args:
            frame (numpy.ndarray): Frame completo.
            gray (numpy.ndarray): Imagen facial en escala de grises del frame.
            x (int): Coordenada x del rectángulo en el que se encuentra la cara detectada.
            y (int): Coordenada y del rectángulo en el que se encuentra la cara detectada.
            w (int): Ancho del rectángulo en el que se encuentra la cara detectada.
            h (int): Alto del rectángulo en el que se encuentra la cara detectada.
            face_cascade (cv2.CascadeClassifier): Clasificador de caras.

        Returns:
            numpy.ndarray: Cara alineada en el rectángulo del frame con un tamaño de 200x200 píxeles.

        """
        alignedImage, angle = self.align_face(frame, gray, x, y, w, h)
        if angle is None:
            return cv2.resize(frame[y:y+h, x:x+w], (200, 200))

        alignedGray = cv2.cvtColor(alignedImage, cv2.COLOR_BGR2GRAY)
        detected_face = face_cascade.detectMultiScale(alignedGray, 1.255, 4)
        if len(detected_face) == 0:
            return cv2.resize(frame[y:y+h, x:x+w], (200, 200))
        xalign = detected_face[0][0]
        yalign = detected_face[0][1]
        walign = detected_face[0][2]
        halign = detected_face[0][3]

        rotatedFace = alignedImage[yalign:yalign+halign, xalign:xalign+walign]
        return cv2.resize(rotatedFace, (200, 200))

    def process_video(self, video_path, label, index):
        """
        Procesa un video y extrae las caras.

        Args:
            video_path (str): Ruta del video.
            label (int): Etiqueta del video (0 para real, 1 para fake).
            index (int): Índice del video.

        Returns:
            list: Lista de caras extraídas.
            list: Lista de etiquetas correspondientes a las caras extraídas.
            int: Índice del video.

        """
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = []
        labels = []
        cap = cv2.VideoCapture(video_path)

        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                if (label == 1 and random.random() < self.percentageExtractionFake) or (label == 0 and random.random() < self.percentageExtractionReal):
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    detected_faces = face_cascade.detectMultiScale(gray, 1.255, 3, minSize=(50, 50))
                    for (x, y, w, h) in detected_faces:
                        alignedFaceImage = self.align_frame(frame, gray[y:y+h, x:x+w], x, y, w, h, face_cascade)
                        faces.append(alignedFaceImage)
                        labels.append(label)
            else:
                break
        cap.release()
        return faces, labels, index

    def process_image(self, image_path, label):
        """
        Procesa una imagen y extrae las caras.

        Args:
            image_path (str): Ruta de la imagen.
            label (int): Etiqueta de la imagen (0 para real, 1 para fake).

        Returns:
            list: Lista de caras extraídas.
            list: Lista de etiquetas correspondientes a las caras extraídas.

        """
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = []
        labels = []
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detected_faces = face_cascade.detectMultiScale(gray, 1.305, 7)
        for (x, y, w, h) in detected_faces:
            alignedFaceImage = self.align_frame(img, gray[y:y+h, x:x+w], x, y, w, h, face_cascade)
            faces.append(alignedFaceImage)
            labels.append(label)

        return faces, labels

    def process_frame_chunk(self, chunk):
        """
        Procesa un fragmento de frames y extrae las caras.

        Args:
            chunk (tuple): tupla con los valores (indice, chunk) 

        Returns:
            list: Lista de tuplas que contienen el índice y el frame original.
            list: Lista de tuplas que contienen el índice y la cara procesada.

        """
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        processed_faces = []
        original_frames = []

        for frame in chunk[1]:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected_faces = face_cascade.detectMultiScale(gray, 1.305, 7)
            if len(detected_faces) > 0:
                x, y, w, h = detected_faces[0]
                alignedFaceImage = self.align_frame(frame, gray[y:y+h], x, y, w, h, face_cascade)
                processed_faces.append(alignedFaceImage)
                original_frames.append(frame)

        return chunk[0], original_frames, processed_faces

    def process_video_to_predict(self, video_path, sequenceLength=None):
        """
        Procesa un video para predecir y extrae las caras.

        Args:
            video_path (str): Ruta del video.
            sequenceLength (int, optional): Longitud de las secuencias de frames. Si es None, se devuelve un solo array de frames.

        Returns:
            numpy.ndarray: Array de frames originales.
            numpy.ndarray: Array de caras procesadas.

        """
        cap = cv2.VideoCapture(video_path)
        frames = []

        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
            else:
                break
        cap.release()
        
        num_chunks = max(max(1, int(math.log(len(frames) + 1))),8)
        
        chunk_size = len(frames) // num_chunks  # Compute the chunk size

        frame_chunks_with_indices = []
        for i in range(num_chunks):
            start_index = i * chunk_size
            end_index = (i + 1) * chunk_size if i < num_chunks - 1 else len(frames)  # Last chunk takes the remaining frames
            chunk = frames[start_index:end_index]
            frame_chunks_with_indices.append((i, chunk))

        processed_faces = []
        original_frames = []

        with ThreadPoolExecutor(max_workers=num_chunks) as executor:
            futures = [executor.submit(self.process_frame_chunk, chunk) for chunk in frame_chunks_with_indices]
            for future in futures:
                index, original_chunk, processed_chunk = future.result()
                for i in range(len(original_chunk)):
                    original_frames.append(original_chunk[i])
                    processed_faces.append(processed_chunk[i])

        if sequenceLength is None:
            return np.array(original_frames), np.array(processed_faces)
        else:
            original_sequences = [original_frames[i:i+sequenceLength] for i in range(0, len(original_frames), sequenceLength)]
            processed_sequences = [processed_faces[i:i+sequenceLength] for i in range(0, len(processed_faces), sequenceLength)]

            if original_sequences and len(original_sequences[-1]) < sequenceLength:
                original_sequences.pop()
                processed_sequences.pop()

            return np.array(original_sequences), np.array(processed_sequences)

    def transform(self, videos, videoLabels):
        """
        Transforma los videos y extrae las caras.

        Args:
            videos (list): Lista de rutas de los videos.
            videoLabels (list): Lista de etiquetas de los videos.

        Returns:
            list: Lista de listas de caras extraídas.
            list: Lista de listas de etiquetas correspondientes a las caras extraídas.

        """
        num_videos = len(videos)
        videoFaces = [[] for i in range(num_videos)]
        groupedLabels = [[] for i in range(num_videos)]
        current = 1
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.process_video, videos[i], videoLabels[i], i): i for i in range(num_videos)}
            for future in concurrent.futures.as_completed(futures):
                print(f'Completed processing of video {current}/{num_videos} ----> {math.floor(current/num_videos*100)}%')
                result = future.result()
                #Guardamos en el index del video procesado
                videoFaces[result[2]].extend(result[0])
                groupedLabels[result[2]].extend(result[1])
                current += 1
        return videoFaces,groupedLabels
    

    def align_face(self, faceImg, greyImg , x,y,w,h):
        """
        Recibe la imagen completa y el rectangulo en el que se encuentra la cara detectada, devuelve el recorte con la cara
        alineada de tal forma que los ojos estén en la misma línea horizontal y la boca en la misma línea vertical
        Si no es capaz de detectar los ojos o de encontrar la cara una vez alineada la imagen devuelve la imagen sin alinear

        :param faceImg: Imagen de la cara.
        :type faceImg: numpy.ndarray
        :param greyImg: Imagen en escala de grises.
        :type greyImg: numpy.ndarray
        :param x: Coordenada x de la esquina superior izquierda del rectángulo de la cara.
        :type x: int
        :param y: Coordenada y de la esquina superior izquierda del rectángulo de la cara.
        :type y: int
        :param w: Ancho del rectángulo de la cara.
        :type w: int
        :param h: Altura del rectángulo de la cara.
        :type h: int
        :return: La imagen de la cara alineada y el ángulo de rotación.
        :rtype: Tuple[numpy.ndarray, float]
        """
        
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

