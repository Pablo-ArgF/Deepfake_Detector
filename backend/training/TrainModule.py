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
from keras.applications import VGG16

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
#Ejemplo de entrenamiento de una CNN como la resultante
inputs = layers.Input(shape=(200, 200, 3))

x = Lambda(lambda x: x/255.0)(inputs)

# Conv1_1 and Conv1_2 Layers
x = conv_prelu(32, (3, 3), 'conv1_1', kernel_regularizer=l2(0.01))(x)
x = conv_prelu(32, (3, 3), 'conv1_2', kernel_regularizer=l2(0.01))(x)
x = layers.Dropout(0.25)(x)  # Adding dropout after Conv1_2

# Pool1 Layer
x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)

# Conv2_1 and Conv2_2 Layers
x = conv_prelu(64, (3, 3), 'conv2_1', kernel_regularizer=l2(0.01))(x)
x = conv_prelu(64, (3, 3), 'conv2_2', kernel_regularizer=l2(0.01))(x)
x = layers.Dropout(0.25)(x)  # Adding dropout after Conv2_2

# Pool2 Layer
pool2_output = layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)

# Additional Convolutional Layers with L2 regularization and Dropout
conv3_1 = conv_prelu(128, (3, 3), 'conv3_1', kernel_regularizer=l2(0.01))(pool2_output)
conv3_2 = conv_prelu(64, (3, 3), 'conv3_2', kernel_regularizer=l2(0.01))(conv3_1)
conv3_2 = layers.Dropout(0.25)(conv3_2)  # Adding dropout after Conv3_2
pool3 = layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(conv3_2)

conv4_1 = conv_prelu(128, (3, 3), 'conv4_1', kernel_regularizer=l2(0.01))(pool3)
conv4_2 = conv_prelu(64, (3, 3), 'conv4_2', kernel_regularizer=l2(0.01))(conv4_1)
conv4_2 = layers.Dropout(0.25)(conv4_2)  # Adding dropout after Conv4_2

conv5_2 = conv_prelu(128, (3, 3), 'conv5_2', kernel_regularizer=l2(0.01))(pool2_output)
conv5_3 = conv_prelu(64, (3, 3), 'conv5_3', kernel_regularizer=l2(0.01))(conv5_2)
conv5_3 = layers.Dropout(0.25)(conv5_3)  # Adding dropout after Conv5_3

conv5_1 = conv_prelu(128, (3, 3), 'conv5_1', kernel_regularizer=l2(0.01))(pool2_output)
concat_1 = layers.Concatenate(name="concat_1")([conv3_2, conv5_1, conv5_3])
pool5 = layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool5')(concat_1)

conv6_2 = conv_prelu(128, (3, 3), 'conv6_2', kernel_regularizer=l2(0.01))(pool5)
conv6_3 = conv_prelu(64, (3, 3), 'conv6_3', kernel_regularizer=l2(0.01))(conv6_2)
conv6_3 = layers.Dropout(0.25)(conv6_3)  # Adding dropout after Conv6_3

conv6_1 = conv_prelu(128, (3, 3), 'conv6_1', kernel_regularizer=l2(0.01))(pool5)
concat_2 = layers.Concatenate(name="concat_2")([conv4_2, conv6_1, conv6_3])

pool4 = layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(concat_2)

flatten = Flatten()(pool4)

fc = layers.Dense(512, name='fc', kernel_regularizer=l2(0.01))(flatten)
fc = layers.Dropout(0.5)(fc)  # Adding dropout before the fully connected layer

fc_class = layers.Dense(1024, name='fc_class', kernel_regularizer=l2(0.01))(fc)
fc_class = layers.Dropout(0.3)(fc_class)
fc_class = layers.Dense(1024, name='fc2', kernel_regularizer=l2(0.01))(fc_class)

# Softmax Output Layer
outputs = layers.Dense(1, activation='sigmoid', name='out')(fc_class)

# Compile the model (add optimizer, loss function, etc.)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Add the model to the training module
trainingModule = TrainModule(route, resultsPath)
trainingModule.addModel(model, 'CNN Model')
trainingModule.startTraining(epochs=2, batchSize=5)
"""


"""
#Ejemplo de entrenamiento de una RNN como la resultante

# Directories for data and results
route = 'P:\\TFG\\Datasets\\dataframes_small'
resultsPath = 'P:\\TFG\\Datasets\\dataframes_small\\results'
routeServer = '/home/pabloarga/Data/dataframes'
sequencesServer = '/home/pabloarga/Data/sequences_20'
resultsPathServer = '/home/pabloarga/Results'

# PReLU alpha value
value_PReLU = 0.25

def conv_prelu(filters, kernel_size, name):
    conv_layer = layers.Conv2D(filters, kernel_size, padding='same', name=name)
    prelu_layer = PReLU(alpha_initializer=Constant(value=value_PReLU))
    bn_layer = layers.BatchNormalization()
    return Sequential([conv_layer, bn_layer, prelu_layer])

# Load the pre-trained CNN model
selected_cnn_model = '2024-05-08 03.07.29'
cnn_model_path = f"/home/pabloarga/Results/{selected_cnn_model}/model{selected_cnn_model}.keras"
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

# Add the LSTM layer with dropout and regularization
lstm_output = LSTM(256, dropout=0.5, recurrent_dropout=0.5, kernel_regularizer=l2(0.001))(cnn_output)

# Dense layers with dropout and regularization
dense_output = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(lstm_output)
dense_output = Dropout(0.3)(dense_output)
dense_output = Dense(32, activation='relu', kernel_regularizer=l2(0.001))(dense_output)
dense_output = Dropout(0.5)(dense_output)

# Final output layer for binary classification
output = Dense(1, activation='sigmoid')(dense_output)

# Create the model
model2 = Model(inputs=cnn_input, outputs=output)
model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Add the model to the training module
trainingModule = TrainModule(route, resultsPath)
trainingModule.addModel(model2, 'RNN Model', isSequence=True)
trainingModule.startTraining(epochs=3, batchSize=3)
"""
