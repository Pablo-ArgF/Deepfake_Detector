# Import the necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
#Para monitorear el uso de CPU y RAM
import threading
import psutil 
import time 

class TrainingMetrics():
    def __init__(self,model,doPlot = True, doConfusionMatrix = True, doSaveResults =True) -> None:
        #Modelo a entrenar
        self.model = model
        #Boolean indicando si se desea hacer un plot de los resultados de entrenamiento
        self.doPlot = doPlot
        #Boolean indicando si se desea enseñar un gráfico con los datos de confusion matrix
        self.doConfusionMatrix = doConfusionMatrix
        #Boolean indicando si se desea guardar los resultados en un archivo csv
        self.doSaveResults = doSaveResults
        #Arrays conteniendo los valores de consumo del entrenamiento del modelo
        self.cpu_percentages = []
        self.memory_percentages = []
        #Boolean indicando si el monitoreo de CPU y RAM está activo
        self.monitoring = False

    """
    Metodo para ser llamado en un hilo paralelo que monitorea el uso de CPU y memoria RAM
    """
    def monitor_usage(self):
        self.monitoring = True
        while self.monitoring:
            self.cpu_percentages.append(psutil.cpu_percent())
            self.memory_percentages.append(psutil.virtual_memory().percent)
            time.sleep(1)  # Paramos la thread durante 1seg para tomar las mediciones cada segundo

    def train(self,X_train,y_train,X_test,y_test,epochs):
        #Defino un hilo que va a estar en paralelo tomando datos de consumo de CPU y memoria RAM
        #además del hilo encargado del entrenamiento del modelo
        print('Started training....')

        monitor_thread = threading.Thread(target=self.monitor_usage)
        monitor_thread.start()

        #Registramos el tiempo que tarda en entrenar
        start_time = time.time()
        self.history_dict = self.model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test))
        end_time = time.time()

        # Paramos la thread de monitoreo
        self.monitoring = False
        self.trainTime = end_time - start_time
        monitor_thread.join()  # Esperamos a que la thread de monitoreo pare

        print(f'Training time: {self.trainTime} seconds')
        print(f'CPU usage: {np.mean(self.cpu_percentages)}%')
        print(f'Memory usage: {np.mean(self.memory_percentages)}%')

        if(self.doPlot):
            self.plot()
        if(self.doConfusionMatrix):
            y_pred = self.model.predict(X_test)
            self.confusionMatrix(y_test, y_pred)
        if(self.doSaveResults):
            self.saveStats()

        return self.model

        

    
    def plot(self):
        print(self.history_dict.keys())
        loss_values = self.history_dict["loss"]
        val_loss_values = self.history_dict["val_loss"]
        epochs = range(1, len(loss_values) + 1)
        plt.figure()
        plt.plot(epochs, loss_values, "bo", label="Pérdida de entrenamiento (training loss)")
        plt.plot(epochs, val_loss_values, "b", label="Pérdida de validación (validation loss)")
        plt.title("Pérdida de entrenamiento y validación")
        plt.xlabel("Épocas")
        plt.ylabel("Pérdida")
        plt.legend()
        plt.show()
        return

    def confusionMatrix(self, real_y, predicted_y):
        # Create a confusion matrix
        conf_matrix = confusion_matrix(real_y, predicted_y)

        # Plot the confusion matrix as a heatmap
        plt.figure(figsize=(5, 5))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()
        return

    def saveStats(self, filePath = "metrics.csv"):
        df = pd.DataFrame(self.history_dict)
        #TODO meter la performance
        df.to_csv(filePath, index=True)






from sklearn.utils import shuffle
from MetricsModule import TrainingMetrics
import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, MaxPooling2D, Conv2D,MaxPooling3D, Conv3D, Flatten
from tensorflow.keras.models import Sequential
from keras.models import load_model
from faceRecon import FaceExtractorMultithread, FaceExtractor


print('Loading data...')
fragments =[ pd.read_hdf(f'dataframes/CelebDB/dataframe{i}_600videos.h5', key=f'df{i}') for i in range(2)]#6
df = pd.concat(fragments)
df = shuffle(df)

print(df.describe())


X = df.drop(['label'], axis = 1)
y = df['label']

print('Dividing dataset into train and test...')
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42, stratify = y)
X_train = np.stack(X_train['face'], axis=0)
X_test = np.stack(X_test['face'], axis=0)



print('Creating model...')
model = Sequential()
model.add(Input(shape=(64, 64, 3)))
model.add(Conv2D(64, (10, 10), activation='relu'))
model.add(MaxPooling2D((6, 6)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='softmax'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

metrics = TrainingMetrics(model,False,False,False)
metrics.train(X_train, y_train, X_test, y_test, epochs=2)