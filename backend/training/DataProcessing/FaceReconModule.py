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
        # Reuse cascades to avoid overhead
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

    def align_frame(self, frame, gray, x, y, w, h, face_cascade, is_pre_detected=False):
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
            is_pre_detected (bool): Indica si la cara ya ha sido detectada previamente.

        Returns:
            numpy.ndarray: Cara alineada en el rectángulo del frame con un tamaño de 200x200 píxeles.

        """
        # If we already have a detection, align_face will use it
        # Note: align_face expects the full gray image, not just the face region
        alignedImage, angle = self.align_face(frame, gray, x, y, w, h, self.eye_cascade)
        if angle is None:
            # Fallback direct resize if alignment fails
            return cv2.resize(frame[y:y+h, x:x+w], (200, 200))

        # Re-detecting ONLY if we were not sure about the initial detection or if we want to refine
        # However, for speed, we can skip detection on the aligned image if we trust the initial alignment
        # Let's refine only if needed. For prediction, speed is key.
        if is_pre_detected:
            # Just crop from the aligned image at the original relative position
            # Since the image was rotated around the face center, the face should be roughly at the same place
            # but better centered.
            # To be safe and precise, we do a quick detection on the aligned image
            # but with restrictive parameters to be fast.
            alignedGray = cv2.cvtColor(alignedImage, cv2.COLOR_BGR2GRAY)
            detected_face = face_cascade.detectMultiScale(alignedGray, 1.255, 4, minSize=(min(w,h)//2, min(w,h)//2))
            if len(detected_face) == 0:
                # If re-detection fails on aligned image, fall back to original crop
                return cv2.resize(frame[y:y+h, x:x+w], (200, 200))
            xf, yf, wf, hf = detected_face[0]
            rotatedFace = alignedImage[yf:yf+hf, xf:xf+wf]
            return cv2.resize(rotatedFace, (200, 200))
        else:
            # Original logic for non-pre-detected faces (e.g., for training data)
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
        """
        faces = []
        labels = []
        cap = cv2.VideoCapture(video_path)

        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                if (label == 1 and random.random() < self.percentageExtractionFake) or (label == 0 and random.random() < self.percentageExtractionReal):
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    detected_faces = self.face_cascade.detectMultiScale(gray, 1.255, 3, minSize=(50, 50))
                    for (x, y, w, h) in detected_faces:
                        # Pass the full gray image to align_frame
                        alignedFaceImage = self.align_frame(frame, gray, x, y, w, h, self.face_cascade)
                        faces.append(alignedFaceImage)
                        labels.append(label)
            else:
                break
        cap.release()
        return faces, labels, index

    def process_image(self, image_path, label):
        """
        Procesa una imagen y extrae las caras.
        """
        faces = []
        labels = []
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detected_faces = self.face_cascade.detectMultiScale(gray, 1.305, 7)
        for (x, y, w, h) in detected_faces:
            alignedFaceImage = self.align_frame(img, gray[y:y+h, x:x+w], x, y, w, h, self.face_cascade)
            faces.append(alignedFaceImage)
            labels.append(label)

        return faces, labels

    def process_frame_chunk(self, chunk):
        """
        Procesa un fragmento de frames y extrae las caras de forma optimizada.
        """
        processed_faces = []
        original_frames = []

        for frame in chunk[1]:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Use faster detection parameters: larger scaleFactor and minSize
            detected_faces = self.face_cascade.detectMultiScale(gray, 1.4, 5, minSize=(64, 64))
            if len(detected_faces) > 0:
                x, y, w, h = detected_faces[0]
                # Pass the full gray image and is_pre_detected=True
                alignedFaceImage = self.align_frame(frame, gray, x, y, w, h, self.face_cascade, is_pre_detected=True)
                processed_faces.append(alignedFaceImage)
                original_frames.append(frame)

        return chunk[0], original_frames, processed_faces

    def process_video_to_predict(self, video_path, unique_id=None, sequenceLength=None, skip_frames=0, save_original_frames=True):
        """
        Procesa un video de forma optimizada para predecir, ahorrando memoria mediante streaming.
        
        Args:
            skip_frames (int): Number of frames to skip after each processed frame to speed up (stride).
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return np.array([]), np.array([])

        processed_faces = []
        
        # Internal counter to keep track of frames saved
        frame_idx = 0
        input_frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Stride logic for speed
            if skip_frames > 0 and input_frame_idx % (skip_frames + 1) != 0:
                input_frame_idx += 1
                continue
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Faster detection
            detected_faces = self.face_cascade.detectMultiScale(gray, 1.4, 5, minSize=(64, 64))
            
            if len(detected_faces) > 0:
                x, y, w, h = detected_faces[0]
                alignedFace = self.align_frame(frame, gray, x, y, w, h, self.face_cascade, is_pre_detected=True)
                
                # We store only the small face in memory
                processed_faces.append(alignedFace)
                
                if unique_id:
                    base_path = f"static/images/generated/{unique_id}"
                    os.makedirs(base_path, exist_ok=True)
                    
                    if sequenceLength:
                        seq_idx = frame_idx // sequenceLength
                        within_seq_idx = frame_idx % sequenceLength
                        non_proc_name = f"sequence_{seq_idx}_frame_{within_seq_idx}.png"
                        proc_name = f"processed_sequence_{seq_idx}_frame_{within_seq_idx}.png"
                    else:
                        non_proc_name = f"nonProcessed_frame_{frame_idx}.jpg"
                        proc_name = f"processed_frame_{frame_idx}.jpg"
                        
                    if save_original_frames:
                        cv2.imwrite(f"{base_path}/{non_proc_name}", frame)
                    cv2.imwrite(f"{base_path}/{proc_name}", alignedFace)
                
                frame_idx += 1
            input_frame_idx += 1

        cap.release()
        
        if len(processed_faces) == 0:
            return np.array([]), np.array([])

        if sequenceLength is None:
            return None, np.array(processed_faces)
        else:
            processed_sequences = [processed_faces[i:i+sequenceLength] for i in range(0, len(processed_faces), sequenceLength)]
            if processed_sequences and len(processed_sequences[-1]) < sequenceLength:
                processed_sequences.pop()
            
            return None, np.array(processed_sequences)

    def stream_video_to_predict(self, video_path, unique_id=None, sequenceLength=None, skip_frames=0, save_original_frames=False):
        """
        Generador que procesa un video y devuelve secuencias de una en una para ahorrar memoria.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return

        current_sequence_faces = []
        frame_idx = 0
        input_frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if skip_frames > 0 and input_frame_idx % (skip_frames + 1) != 0:
                input_frame_idx += 1
                continue
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected_faces = self.face_cascade.detectMultiScale(gray, 1.4, 5, minSize=(64, 64))
            
            if len(detected_faces) > 0:
                x, y, w, h = detected_faces[0]
                alignedFace = self.align_frame(frame, gray, x, y, w, h, self.face_cascade, is_pre_detected=True)
                
                if unique_id:
                    base_path = f"static/images/generated/{unique_id}"
                    os.makedirs(base_path, exist_ok=True)
                    
                    if sequenceLength:
                        seq_idx = frame_idx // sequenceLength
                        within_seq_idx = frame_idx % sequenceLength
                        non_proc_name = f"sequence_{seq_idx}_frame_{within_seq_idx}.png"
                        proc_name = f"processed_sequence_{seq_idx}_frame_{within_seq_idx}.png"
                    else:
                        non_proc_name = f"nonProcessed_frame_{frame_idx}.jpg"
                        proc_name = f"processed_frame_{frame_idx}.jpg"
                    
                    if save_original_frames:
                        cv2.imwrite(f"{base_path}/{non_proc_name}", frame)
                    cv2.imwrite(f"{base_path}/{proc_name}", alignedFace)
                
                if sequenceLength:
                    current_sequence_faces.append(alignedFace)
                    if len(current_sequence_faces) == sequenceLength:
                        yield np.array(current_sequence_faces)
                        current_sequence_faces = []
                else:
                    yield alignedFace
                
                frame_idx += 1
            input_frame_idx += 1

        cap.release()

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
    

    def align_face(self, faceImg, greyImg , x,y,w,h, eye_detector):
        """
        Recibe la imagen completa y el rectangulo en el que se encuentra la cara detectada, devuelve el recorte con la cara
        alineada de tal forma que los ojos estén en la misma línea horizontal y la boca en la misma línea vertical
        Si no es capaz de detectar los ojos o de encontrar la cara una vez alineada la imagen devuelve la imagen sin alinear
        """
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

