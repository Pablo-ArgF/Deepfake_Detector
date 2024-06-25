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

# Directories for data and results
route = 'P:\\TFG\\Datasets\\dataframes_small'
resultsPath = 'P:\\TFG\\Datasets\\dataframes_small\\results'
routeServer = '/home/pabloarga/Data/dataframes'
sequencesServer = '/home/pabloarga/Data/sequences_20'
resultsPathServer = '/home/pabloarga/Results'

# PReLU alpha value
value_PReLU = 0.25

def conv_prelu(filters, kernel_size, name, kernel_regularizer=None):
    conv_layer = layers.Conv2D(filters, kernel_size, padding='same', name=name, kernel_regularizer=kernel_regularizer)
    prelu_layer = PReLU(alpha_initializer=Constant(value=value_PReLU))
    bn_layer = layers.BatchNormalization()
    return Sequential([conv_layer, bn_layer, prelu_layer])

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

models = {
    model: "RNN using LSTM, pretrained CNN, dense layers, and overfitting prevention techniques",
}

for model, description in models.items():
    metrics = TrainingMetrics(model, resultsPathServer, modelDescription=description)
    metrics.batches_train(folderPath=sequencesServer, nPerBatch=4, epochs=4, isSequence=True)
