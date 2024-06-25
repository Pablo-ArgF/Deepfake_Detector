import os
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import Dense, MaxPooling2D, Conv2D, Flatten, Input, Dropout, PReLU, BatchNormalization, LSTM, TimeDistributed
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from MetricsModule import TrainingMetrics
import numpy as np
import h5py

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class TrainModule():
    """
    Clase utilizada para gestionar el proceso de entrenamiento de modelos.

    :param routeToData: La ruta al conjunto de datos.
    :type routeToData: str
    :param routeToResults: La ruta donde se guardarán los resultados.
    :type routeToResults: str
    :ivar models: Una lista de modelos para ser entrenados.
    :vartype models: list
    :ivar descriptions: Una lista de descripciones para los modelos.
    :vartype descriptions: list
    :ivar isSequence: Una lista que indica si el modelo correspondiente espera datos secuenciales.
    :vartype isSequence: list

    :method conv_prelu(filters, kernel_size, name, kernel_regularizer=None): Crea una capa convolucional seguida de una activación PReLU y normalización por lotes.
    :method addModel(model, description, isSequence=False): Agrega un modelo al módulo de entrenamiento.
    :method removeModel(model): Elimina un modelo del módulo de entrenamiento.
    :method startTraining(epochs, batchSize): Inicia el proceso de entrenamiento para todos los modelos agregados.
    """

    def __init__(self, routeToData, routeToResults) -> None:
        """
        Parámetros
        ----------
        routeToData : str
            La ruta al conjunto de datos.
        routeToResults : str
            La ruta donde se guardarán los resultados.
        """
        self.routeToData = routeToData
        self.routeToResults = routeToResults
        self.models = []
        self.descriptions = []
        self.isSequence = []

    @staticmethod
    def conv_prelu(filters, kernel_size, name, kernel_regularizer=None):
        """
        Crea una capa convolucional seguida de una activación PReLU y normalización por lotes.
    
        :param filters: El número de filtros en la capa convolucional.
        :type filters: int
        :param kernel_size: El tamaño del kernel en la capa convolucional.
        :type kernel_size: tuple
        :param name: El nombre de la capa convolucional.
        :type name: str
        :param kernel_regularizer: La función de regularización aplicada a la matriz de pesos del kernel, opcional.
        :type kernel_regularizer: regularizer
        :return: Un modelo secuencial de Keras que consiste en las capas Conv2D, BatchNormalization y PReLU.
        :rtype: Sequential
        """
        value_PReLU = 0.25
        conv_layer = layers.Conv2D(filters, kernel_size, padding='same', name=name, kernel_regularizer=kernel_regularizer)
        prelu_layer = PReLU(alpha_initializer=Constant(value=value_PReLU))
        bn_layer = layers.BatchNormalization()
        return Sequential([conv_layer, bn_layer, prelu_layer])
        
    def addModel(self, model, description, isSequence=False):
        """
        Agrega un modelo al módulo de entrenamiento.
    
        :param model: El modelo a agregar.
        :type model: keras.Model
        :param description: Una breve descripción del modelo.
        :type description: str
        :param isSequence: Indica si el modelo espera datos secuenciales, opcional.
        :type isSequence: bool
        """
        self.models.append(model)
        self.descriptions.append(description)
        self.isSequence.append(isSequence)

    def removeModel(self, model):
        """
        Elimina un modelo del módulo de entrenamiento.
    
        :param model: El modelo a eliminar.
        :type model: keras.Model
        """
        index = self.models.index(model)
        self.models.pop(index)
        self.descriptions.pop(index)
        self.isSequence.pop(index)

    def startTraining(self, epochs, batchSize):
        """
        Inicia el proceso de entrenamiento para todos los modelos agregados.
    
        :param epochs: El número de épocas para el entrenamiento.
        :type epochs: int
        :param batchSize: El tamaño del lote para el entrenamiento.
        :type batchSize: int
        """
        for model, description, isSequence in zip(self.models, self.descriptions, self.isSequence):
            metrics = TrainingMetrics(model, self.routeToResults, modelDescription=description)
            metrics.batches_train(folderPath=self.routeToData, nPerBatch=batchSize, epochs=epochs, isSequence=isSequence)

"""
# Directories for data and results
route = 'P:\\TFG\\Datasets\\dataframes_small'
resultsPath = 'P:\\TFG\\Datasets\\dataframes_small\\results'
routeServer = '/home/pabloarga/Data/dataframes'
sequencesServer = '/home/pabloarga/Data/sequences_20'
resultsPathServer = '/home/pabloarga/Results'



selectedCNN = '2024-06-20 08.29.00'
cnn_model_path = f"/home/pabloarga/Results/{selectedCNN}/model{selectedCNN}.keras"
cnn_base = tf.keras.models.load_model(cnn_model_path, safe_mode=False, compile=False)

# Ensure the CNN layers are not trainable
cnn_base.trainable = False

# Define the input shape for the RNN
input_shape = (20, 200, 200, 3)
cnn_input = Input(shape=input_shape)

# Wrap the CNN in a TimeDistributed layer to process sequences of images
cnn_output = TimeDistributed(cnn_base)(cnn_input)
cnn_output = TimeDistributed(Flatten())(cnn_output)
cnn_output = TimeDistributed(Dropout(0.5))(cnn_output)  # Adding dropout after the CNN

# Add the LSTM layer with dropout, batch normalization, and regularization
lstm_output = LSTM(256, dropout=0.5, recurrent_dropout=0.5, kernel_regularizer=l2(0.001), return_sequences=True)(cnn_output)
lstm_output = BatchNormalization()(lstm_output)
lstm_output = LSTM(128, dropout=0.5, recurrent_dropout=0.5, kernel_regularizer=l2(0.001))(lstm_output)

# Dense layers with dropout, batch normalization, and regularization
dense_output = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(lstm_output)
dense_output = BatchNormalization()(dense_output)
dense_output = Dropout(0.5)(dense_output)
dense_output = Dense(1024, activation='relu', kernel_regularizer=l2(0.001))(dense_output)
dense_output = BatchNormalization()(dense_output)
dense_output = Dropout(0.5)(dense_output)

# Final output layer for binary classification
output = Dense(1, activation='sigmoid')(dense_output)

# Create the model
model = Model(inputs=cnn_input, outputs=output)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Add the model to the training module
trainModule = TrainModule(routeServer,resultsPathServer)
trainModule.addModel(model,'RNN using LSTM, pretrained CNN, dense layers, and overfitting prevention techniques',isSequence=True)
trainModule.startTraining(epochs = 3,batchSize = 4)
"""

