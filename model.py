import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
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
from faceRecon import FaceExtractorMultithread, FaceExtractor

def loadData(baseDir):
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
    #dataFrame = dataFrame.sample(1000, random_state=42)

    face_extractor = FaceExtractor(n=100)#, max_workers=5)
    print('Extracting faces from videos...')
    dataFrame= face_extractor.transform(dataFrame)
    return dataFrame


#df = loadData(baseDir='Datasets\CelebDB\Celeb-DF-v2')
#load from files dataframe0...5.h5 to df
df = pd.DataFrame()

fragments =[ pd.read_hdf(f'dataframe{i}.h5', key=f'df{i}') for i in range(6)]
df = pd.concat(fragments)
df = shuffle(df)

print('Dividing dataset into train and test...')
# Dividir el dataset en train y test
X = df.drop(['label'], axis = 1)
y = df['label']


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42, stratify = y)
X_train = np.stack(X_train['face'], axis=0)
X_test = np.stack(X_test['face'], axis=0)



y_train = y_train.astype(float)
y_test = y_test.astype(float)


model = Sequential()
model.add(Input(shape=(64, 64, 3)))
model.add(Conv2D(10, (5, 5), activation='relu'))
model.add(MaxPooling2D((3, 3)))
model.add(Conv2D(20, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(10, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='softmax'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

print('Started training...')

model.fit(X_train, y_train, epochs=60, validation_data=(X_test, y_test))

#evaluamos el modelo
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print('Test Accuracy: %.3f' % acc)
print('Test Loss: %.3f' % loss)

#exportamos el modelo
model.save('model_600videos_n5frames.h5')



