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
def testingDataset(baseDir):
    testing_videos = []
    testing_labels = []
    with open('Datasets\CelebDB\Celeb-DF-v2\List_of_testing_videos.txt', 'r') as file:
        for line in file:
            testing_videos.append(os.path.join(baseDir,line.split()[1]))
            testing_labels.append(line.split()[0])
    dataFrame = pd.DataFrame({'video': testing_videos, 'label': testing_labels})
    return dataFrame

#cargamos los videos el dataset de entrenamiento, para ello iternamos por la carpeta de videos y guardamos en 
# el datasframe el nombre y categoría de aquellos que no han aparecido en el dataset de test
def trainingDataset(baseDir):
    training_videos = []
    training_labels = []
    testingVideos =  testingDataset(baseDir)['video'].values.tolist()
    # Iterate over the folders of videos inside Celeb-DF-v2
    for folder in os.listdir(baseDir):
        folder_path = os.path.join(baseDir, folder)
        if not os.path.isdir(folder_path):
            continue
        for video in os.listdir(folder_path):
            video_path = os.path.join(folder_path, video)
            if video not in testingVideos:
                training_videos.append(video_path)
                if (folder.split('-')[1] == 'real'):
                    training_labels.append(1)
                else:
                    training_labels.append(0)
    dataFrame = pd.DataFrame({'video': training_videos, 'label': training_labels})
    return dataFrame


df_train = trainingDataset(baseDir='Datasets\CelebDB\Celeb-DF-v2')
df_test = testingDataset(baseDir='Datasets\CelebDB\Celeb-DF-v2')



# Reducir el tamaño de los dataframes a 200 filas para entrenamiento y 100 filas para test, de forma reproducible
df_train = df_train.sample(10, random_state=42)
df_test = df_test.sample(5, random_state=42)





face_extractor = FaceExtractor(n=10, output_dir='imgs')
X_train, y_train = face_extractor.transform(df_train)
X_test, y_test = face_extractor.transform(df_test)

y_train = y_train.astype(float)
y_test = y_test.astype(float)


model = Sequential()
model.add(Input(shape=(64, 64, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='softmax'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

#evaluamos el modelo
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print('Test Accuracy: %.3f' % acc)
print('Test Loss: %.3f' % loss)



