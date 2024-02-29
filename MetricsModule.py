# Import the necessary libraries
import os
import gc #Garbage collector for freeing memory
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
#Para monitorear el uso de CPU y RAM
import threading
import psutil 
import time 
import matplotlib as mpl

class TrainingMetrics():
    def __init__(self,model,resultDataPath,showGraphs = False) -> None:   
        mpl.use('Agg')
        #Modelo a entrenar
        self.model = model
        #Path donde se guardan los resultados (imagenes, csv, etc)
        self.resultDataPath = resultDataPath
        #Boolean indicando si se desea enseñar las gráficas de perdida precisioni y la matriz de confusion (aun siendo falsos estos datos se guardaran)
        self.showGraphs = showGraphs
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


    """
    Metodo que recibe el path a una carpeta en la que se encuentran los Dataframes y con ellos
    entrena el modelo. El entrenamiento del modelo se hace cargando N dataframes a la vez, minimizando
    el uso de memoria RAM
    """
    def batches_train(self,folderPath,nBatches,epochs):
        #Obtenemos el numero de dataframes que hay en la carpeta
        numDataframes = len([name for name in os.listdir(folderPath) if os.path.isfile(os.path.join(folderPath, name))])
        #Calculamos el tamaño de cada fragmento
        fragmentSize = int(numDataframes/nBatches)

        #Iteramos por cada batch cargando los dataframes y usandolos para entrenar el modelo
        for i in range(nBatches):
            print(f'Training the model with batch: {i+1}/{nBatches}')
            #Cargamos los dataframes del batch y los guardamos en un solo dataframe
            fragments = [pd.read_hdf(f'{folderPath}/dataframe{j}_FaceForensics.h5', key=f'df{j}') for j in range(fragmentSize*i,fragmentSize*(i+1))]
            df = pd.concat(fragments)


            #Dividimos el dataframe en train y test
            X = np.squeeze(df.drop(['label'], axis = 1)) #Eliminamos la columna de etiquetas y lo dejamos como un vector
            y = df['label'].astype(int)
            X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
            X_train = np.stack(X_train, axis=0)
            X_test = np.stack(X_test, axis=0)

            #Entrenamos el modelo con el dataframe     
            self.train(X_train, y_train, X_test, y_test, epochs)
            
            #Liberamos memoria
            del df
            del fragments
            del X_train
            del X_test
            del y_train
            del y_test
            gc.collect()

    def train(self,X_train,y_train,X_test,y_test,epochs):
        #Defino un hilo que va a estar en paralelo tomando datos de consumo de CPU y memoria RAM
        #además del hilo encargado del entrenamiento del modelo
        #print('Started training....')

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

        """
        print(f'Training time: {self.trainTime} seconds')
        print(f'CPU usage: {np.mean(self.cpu_percentages)}%')
        print(f'Memory usage: {np.mean(self.memory_percentages)}%')
        """

        self.plot()
        y_pred = self.model.predict(X_test)
        # Pasamos de probabilidades a una clasificación en fake o on fake
        y_pred = np.where(y_pred > 0.5, 1, 0)
        self.confusionMatrix(y_test, y_pred)
        self.saveStats()

        return self.history_dict.model

    
    def plot(self):
        # Si en el path no existe una carpeta plots, la creamos
        plotsFolder = os.path.join(self.resultDataPath,'plots')
        if not os.path.exists(plotsFolder):
            os.makedirs(plotsFolder)
        
        loss_values = self.history_dict.history["loss"]
        val_loss_values = self.history_dict.history["val_loss"]
        epochs = range(1, len(loss_values) + 1)
        plt.figure()
        plt.plot(epochs, loss_values, "bo", label="Pérdida de entrenamiento (training loss)")
        plt.plot(epochs, val_loss_values, "b", label="Pérdida de validación (validation loss)")
        plt.title("Pérdida de entrenamiento y validación")
        plt.xlabel("Épocas")
        plt.ylabel("Pérdida")
        plt.legend()
        #Guardamos el archivo en un fichero
        currentTime = time.strftime("%Y-%m-%d %H.%M.%S")
        plt.savefig(os.path.join(plotsFolder,f"loss_{currentTime}.png")) 
        if self.showGraphs:
            plt.show()
        plt.close()
        return

    def confusionMatrix(self, real_y, predicted_y):
        # Si no existe una carpeta para guardar las matrices de confusión, la creamos
        confusionMatrixFolder = os.path.join(self.resultDataPath,'confusionMatrix')
        if not os.path.exists(confusionMatrixFolder):
            os.makedirs(confusionMatrixFolder)

        # Create a confusion matrix
        conf_matrix = confusion_matrix(real_y, predicted_y)

        # Plot the confusion matrix as a heatmap
        plt.figure(figsize=(5, 5))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        #Guardamos el archivo en un fichero
        currentTime = time.strftime("%Y-%m-%d %H.%M.%S")
        plt.savefig(os.path.join(confusionMatrixFolder,f"confusionMatrix_{currentTime}.png")) 
        if self.showGraphs:
            plt.show()
        plt.close()
        return

    def saveStats(self, fileName = "metrics.csv"):
        filePath = os.path.join(self.resultDataPath,fileName)
        df = pd.DataFrame()
        #Añadimos fecha y hora de estos datos
        currentTime = time.strftime("%Y-%m-%d %H.%M.%S")
        df['date'] = [currentTime] 
        df['trainTime'] = [self.trainTime]
        df['cpuUsage'] = [np.mean(self.cpu_percentages)]
        df['memoryUsage'] = [np.mean(self.memory_percentages)]
        df['epochs'] = [self.history_dict.params['epochs']]
        df['loss'] = [self.history_dict.history['loss'][-1]]
        df['accuracy'] = [self.history_dict.history['accuracy'][-1]]
        #Si no existe el archivo, lo creamos con las cabezeras
        if not os.path.isfile(filePath):
            df.to_csv(filePath, index=False)
        else: #Si ya existe, añadimos la fila al final
            df.to_csv(filePath, mode='a', header=False, index=False)
