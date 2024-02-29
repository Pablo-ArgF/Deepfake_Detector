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
        # Concatenamos la fecha y hora actual al valor de resultDataPath para que cada vez que se ejecute el script se cree una nueva carpeta
        self.baseDir = resultDataPath
        self.currentTime = time.strftime("%Y-%m-%d %H.%M.%S")
        resultDataPath = os.path.join(resultDataPath,self.currentTime)
        os.makedirs(resultDataPath)
        self.resultDataPath = resultDataPath
        #Boolean indicando si se desea enseñar las gráficas de perdida precisioni y la matriz de confusion (aun siendo falsos estos datos se guardaran)
        self.showGraphs = showGraphs
        #Arrays conteniendo los valores de consumo del entrenamiento del modelo
        self.cpu_percentages = np.array([])
        self.memory_percentages = np.array([])
        #Boolean indicando si el monitoreo de CPU y RAM está activo
        self.monitoring = False
        #Arrays para guardar el historico de perdida y precision
        self.loss_history = np.array([]) 
        self.acc_history = np.array([])
        self.val_loss_history = np.array([])
        self.val_acc_history = np.array([])
        #Guardamos datos para poder hacer una matriz de confusión con todos los datos de test de todos los batches
        self.real_y = np.array([])
        self.predicted_y = np.array([])
        #Añadimos unos contadores de número de imagenes fake y reales
        self.numFakeImages = 0
        self.numRealImages = 0

        #Guardamos una representación del modelo en un archivo txt
        with open(os.path.join(self.resultDataPath,f'model_{self.currentTime}.txt'), 'w') as f:
            self.model.summary(print_fn=lambda x: f.write(x + '\n'))

    """
    Metodo para ser llamado en un hilo paralelo que monitorea el uso de CPU y memoria RAM
    """
    def monitor_usage(self):
        self.monitoring = True
        while self.monitoring:
            self.cpu_percentages = np.append(self.cpu_percentages,psutil.cpu_percent())
            self.memory_percentages = np.append(self.memory_percentages,psutil.virtual_memory().percent)
            time.sleep(5)  # Paramos la thread durante 5seg para tomar las mediciones cada 5seg


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

        #Defino un hilo que va a estar en paralelo tomando datos de consumo de CPU y memoria RAM
        #además del hilo encargado del entrenamiento del modelo
        monitor_thread = threading.Thread(target=self.monitor_usage)
        monitor_thread.start()

        #Registramos el tiempo que tarda en entrenar
        start_time = time.time()

        #Iteramos por cada batch cargando los dataframes y usandolos para entrenar el modelo
        for i in range(nBatches):
            print(f'Training the model with batch: {i+1}/{nBatches}')
            #Cargamos los dataframes del batch y los guardamos en un solo dataframe
            fragments = [pd.read_hdf(f'{folderPath}/dataframe{j}_FaceForensics.h5', key=f'df{j}') for j in range(fragmentSize*i,fragmentSize*(i+1))]
            df = pd.concat(fragments)


            #Dividimos el dataframe en train y test
            X = np.squeeze(df.drop(['label'], axis = 1)) #Eliminamos la columna de etiquetas y lo dejamos como un vector
            y = df['label'].astype(int)
            
            #Contamos el número de imagenes fake y reales
            self.numFakeImages += y[y == 0].size
            self.numRealImages += y[y == 1].size

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

        # Paramos la thread de monitoreo
        self.monitoring = False
        end_time = time.time()
        self.trainTime = end_time - start_time
        monitor_thread.join()  # Esperamos a que la thread de monitoreo pare


        self.plot()
        self.confusionMatrix()
        self.saveStats()

        #exportamos el modelo
        self.model.save(os.path.join(self.resultDataPath,f'model{self.currentTime}.keras'))



    def train(self,X_train,y_train,X_test,y_test,epochs):
       
        batchHistory = self.model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test))
        #Guardamos el historico de perdida y precision
        self.loss_history = np.append(self.loss_history, batchHistory.history['loss'])
        self.val_loss_history = np.append(self.val_loss_history, batchHistory.history['val_loss'])
        self.acc_history = np.append(self.acc_history, batchHistory.history['accuracy'])
        self.val_acc_history = np.append(self.val_acc_history, batchHistory.history['val_accuracy'])
        

        #Guardamos los datos de test y predicciones para hacer una matriz de confusión con todos los datos
        self.real_y = np.append(self.real_y, y_test)
        self.predicted_y = np.append(self.predicted_y, self.model.predict(X_test))
        


    
    def plot(self):        
        # Plot the training and validation accuracy
        epochs = range(1, self.loss_history.size + 1)
        plt.figure()
        plt.plot(epochs, self.loss_history, "bo", label="Pérdida de entrenamiento (training loss)")
        plt.plot(epochs, self.val_loss_history, "b", label="Pérdida de validación (validation loss)")
        plt.plot(epochs, self.acc_history, "ro", label="Precisión de entrenamiento (training accuracy)")
        plt.plot(epochs, self.val_acc_history, "r", label="Precisión de validación (validation accuracy)")
        plt.title("Pérdida y precisión de entrenamiento y validación")
        plt.xlabel("Épocas")
        plt.ylabel("Pérdida y precisión")
        plt.legend()
        #Guardamos el archivo en un fichero
        currentTime = time.strftime("%Y-%m-%d %H.%M.%S")
        plt.savefig(os.path.join(self.resultDataPath,"plot.png")) 
        if self.showGraphs:
            plt.show()
        plt.close()
        return

    def confusionMatrix(self):
    
        # Create a confusion matrix
        conf_matrix = confusion_matrix(self.real_y, self.predicted_y.round())

        # Plot the confusion matrix as a heatmap
        plt.figure(figsize=(5, 5))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        #Guardamos el archivo en un fichero
        plt.savefig(os.path.join(self.resultDataPath,f"confusionMatrix_{self.currentTime}.png")) 
        if self.showGraphs:
            plt.show()
        plt.close()
        return

    def saveStats(self, fileName = "metrics.csv"):
        filePath = os.path.join(self.baseDir,fileName)
        df = pd.DataFrame()
        #Añadimos fecha y hora de estos datos
        df['date'] = [self.currentTime] 
        df['totalNumberOfImages'] = [self.numFakeImages + self.numRealImages]
        df['numberOfFakeImages'] = [self.numFakeImages]
        df['numberOfRealImages'] = [self.numRealImages]
        df['epochs'] = [self.loss_history.size]
        df['loss'] = [self.loss_history[-1]]
        df['accuracy'] = [self.acc_history[-1]]
        df['trainTime'] = [self.trainTime]
        df['cpuUsage'] = [np.mean(self.cpu_percentages)]
        df['memoryUsage'] = [np.mean(self.memory_percentages)]
        
        #Si no existe el archivo, lo creamos con las cabezeras
        if not os.path.isfile(filePath):
            df.to_csv(filePath, index=False)
        else: #Si ya existe, añadimos la fila al final
            df.to_csv(filePath, mode='a', header=False, index=False)
