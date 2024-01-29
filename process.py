import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, MaxPooling2D, Conv2D, Flatten
from tensorflow.keras.models import Sequential
from keras.models import load_model
from faceRecon import FaceExtractorMultithread, FaceExtractor


baseDir='Datasets\CelebDB\Celeb-DF-v2'
videos = []
labels = []
# Iterate over the folders of videos inside Celeb-DF-v2
for folder in os.listdir(baseDir):
    folder_path = os.path.join(baseDir, folder)
    if not os.path.isdir(folder_path):
        continue
    for video in os.listdir(folder_path):
        video_path = os.path.join(folder_path, video)
        videos.append(video_path)
        if (folder.split('-')[1] == 'real'):
            labels.append(1)
        else:
            labels.append(0)
dataFrame = pd.DataFrame({'video': videos, 'label': labels})

# Reduce el tamaño del dataset para que sea más fácil de manejar
#dataFrame = dataFrame.sample(1200, random_state=42)

face_extractor = FaceExtractorMultithread(n=20)
print('Extracting faces from videos...')
# Dividimos el dataframe en 6 partes para procesarlo mejor en ram
fragmentSize = int(len(dataFrame)/6)
for i in range(6):
    print(f'Processing fragment {i+1}/6')
    processed = face_extractor.transform(dataFrame.iloc[fragmentSize*i : fragmentSize*(i+1)])
    # Guardamos el fragmento procesado en un fichero hdf
    processed.to_hdf(f'.\dataframes\CelebDB\dataframe{i}_{len(dataFrame)}videos.h5', key=f'df{i}', mode='w')
