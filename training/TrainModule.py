import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import Dense, MaxPooling2D, Conv2D, Flatten, Lambda, Input, Dropout, PReLU, BatchNormalization, GRU
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.regularizers import l2
from MetricsModule import TrainingMetrics
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import TimeDistributed, Dense, LSTM, Input, Flatten, Dropout
from tensorflow.keras.models import Model
import os
import numpy as np
import h5py

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

route = 'P:\\TFG\\Datasets\\dataframes_small'
resultsPath = 'P:\\TFG\\Datasets\\dataframes_small\\results'

routeServer = '/home/pabloarga/Data/dataframes'
sequencesServer = '/home/pabloarga/Data/sequences_20'
resultsPathServer = '/home/pabloarga/Results'

value_PReLU = 0.25

def conv_prelu(filters, kernel_size, name):
    conv_layer = layers.Conv2D(filters, kernel_size, padding='same', name=name)
    prelu_layer = PReLU(alpha_initializer=Constant(value=value_PReLU))
    bn_layer = layers.BatchNormalization()
    return Sequential([conv_layer, bn_layer, prelu_layer])


selected_cnn_model = '2024-06-08 18.09.03'
cnn_model_path = f"/home/pabloarga/Results/{selected_cnn_model}/model{selected_cnn_model}.keras"
cnn_base = tf.keras.models.load_model(cnn_model_path, safe_mode=False, compile=False)

# Ensure the CNN layers are not trainable
cnn_base.trainable = False

# Define the input shape for the RNN
input_shape = (20, 200, 200, 3)  # None for sequence length
cnn_input = Input(shape=input_shape)

# Wrap the CNN in a TimeDistributed layer to process sequences of images
cnn_output = TimeDistributed(cnn_base)(cnn_input)
cnn_output = TimeDistributed(Flatten())(cnn_output)
cnn_output = TimeDistributed(Dropout(0.5))(cnn_output)  # Adding dropout after the CNN

# Add the GRU layer
lstm_output = LSTM(256, dropout=0.5, recurrent_dropout=0.5)(cnn_output)

dense_output = Dense(64, activation='relu')(lstm_output)
dense_output = Dropout(0.3)(dense_output)  # Adding dropout after the first Dense layer
dense_output = Dense(32, activation='relu')(dense_output)
dense_output = Dropout(0.5)(dense_output)  # Adding dropout after the second Dense layer

# Final output layer for binary classification
output = Dense(1, activation='sigmoid')(dense_output)

# Create the model
model2 = Model(inputs=cnn_input, outputs=output)
model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])



models = {
    model2: "RNN usando LSTM, la CNN preentrenada 2024-06-08 18.09.03  y dos capas de red neuronal densa a la salida",
}

for model, description in models.items():
    metrics = TrainingMetrics(model, resultsPathServer, modelDescription=description)
    metrics.batches_train(folderPath=sequencesServer, nPerBatch=3, epochs=3, isSequence=True)
