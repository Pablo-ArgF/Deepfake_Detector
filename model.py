import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.datasets import load_breast_cancer, make_circles, make_classification
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import image_dataset_from_directory
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models, Input
from tensorflow.keras.datasets import imdb, mnist, reuters
from tensorflow.keras.layers import Dense, MaxPooling2D, Conv2D, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers
from keras.optimizers import SGD
from keras.utils import plot_model
from keras.models import load_model
from faceRecon import FaceExtractor


#cargamos del archivo List_of_testing_videos.txt los nombres de los videos de test y su categoria
# guardado en el archivo con estructura <0/1 ruta_video>
# 0 -> fake
# 1 -> real
def testingDataset():
    testing_videos = []
    testing_labels = []
    with open('Datasets\CelebDB\Celeb-DF-v2\List_of_testing_videos.txt', 'r') as file:
        for line in file:
            testing_videos.append(line.split()[1])
            testing_labels.append(line.split()[0])
    dataFrame = pd.DataFrame({'video': testing_videos, 'label': testing_labels})
    return dataFrame

#cargamos los videos el dataset de entrenamiento, para ello iternamos por la carpeta de videos y guardamos en 
# el datasframe el nombre y categoría de aquellos que no han aparecido en el dataset de test
def trainingDataset():
    training_videos = []
    training_labels = []
    testingVideos =  testingDataset()['video'].values.tolist()
    #iternamos por las carpetas de videos dentro de Celeb-DF-v2
    for folder in os.listdir('Datasets\CelebDB\Celeb-DF-v2'):
        if(folder == 'List_of_testing_videos.txt'):
            continue
        for video in os.listdir('Datasets\CelebDB\Celeb-DF-v2\\' + folder):
            if video not in testingVideos:
                training_videos.append(video)
                if (folder.split('-')[1] == 'real'):
                    training_labels.append(1)
                else:
                    training_labels.append(0)
    dataFrame = pd.DataFrame({'video': training_videos, 'label': training_labels})
    return dataFrame


df_train = trainingDataset()
df_test = testingDataset()



# Reducir el tamaño de los dataframes a 200 filas para entrenamiento y 100 filas para test, de forma reproducible
df_train = df_train.sample(200, random_state=42)
df_test = df_test.sample(100, random_state=42)





face_extractor = FaceExtractor(n=10)
X_train = face_extractor.transform(df_train['video'])
X_test = face_extractor.transform(df_test['video'])

# Normaliza las imágenes
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# Codifica las etiquetas
y_train = to_categorical(df_train['label'])
y_test = to_categorical(df_test['label'])

model = Sequential()
model.add(Input(shape=(64, 64, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))



