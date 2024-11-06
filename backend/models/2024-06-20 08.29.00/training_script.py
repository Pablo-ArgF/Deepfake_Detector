import os
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import Dense, MaxPooling2D, Conv2D, Flatten, Lambda, Input, Dropout, PReLU, BatchNormalization, GRU, LSTM, TimeDistributed
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.regularizers import l2
from MetricsModule import TrainingMetrics
from tensorflow.keras.models import load_model
import numpy as np
import h5py

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Directories for data and results
route = 'P:\\TFG\\Datasets\\dataframes_small'
resultsPath = 'P:\\TFG\\Datasets\\dataframes_small\\results'
routeServer = '/home/pabloarga/Data/dataframes'
sequencesServer = '/home/pabloarga/Data/sequences_50'
resultsPathServer = '/home/pabloarga/Results'

# PReLU alpha value
value_PReLU = 0.25

def conv_prelu(filters, kernel_size, name, kernel_regularizer=None):
    conv_layer = layers.Conv2D(filters, kernel_size, padding='same', name=name, kernel_regularizer=kernel_regularizer)
    prelu_layer = PReLU(alpha_initializer=Constant(value=value_PReLU))
    bn_layer = layers.BatchNormalization()
    return Sequential([conv_layer, bn_layer, prelu_layer])

#CNN----------------------------------------------------------------------------------------------------------------------
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

models = {
    model: "Improved CNN to reduce overfitting",
}

for model, description in models.items():
    metrics = TrainingMetrics(model, resultsPathServer, modelDescription=description)
    metrics.batches_train(folderPath=routeServer, nPerBatch=5, epochs=3, isSequence=False)
