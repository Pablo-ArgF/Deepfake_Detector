import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import Dense, MaxPooling2D, Conv2D, Flatten, Lambda, Input, Dropout, PReLU, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.regularizers import l2
from MetricsModule import TrainingMetrics

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

route = 'P:\\TFG\\Datasets\\dataframes_small'
resultsPath = 'P:\\TFG\\Datasets\\dataframes_small\\results'

routeServer = '/home/pabloarga/Data/dataframes'
resultsPathServer = '/home/pabloarga/Results'

value_PReLU = 0.25

def conv_prelu(filters, kernel_size, name):
    conv_layer = layers.Conv2D(filters, kernel_size, padding='same', name=name)
    prelu_layer = PReLU(alpha_initializer=Constant(value=value_PReLU))
    bn_layer = layers.BatchNormalization()
    return Sequential([conv_layer, bn_layer, prelu_layer])

inputs = layers.Input(shape=(200, 200, 3))
x = conv_prelu(32, (3, 3), 'conv1_1')(inputs)
x = conv_prelu(32, (3, 3), 'conv1_2')(x)
x = layers.Dropout(0.25)(x)

x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)

x = conv_prelu(64, (3, 3), 'conv2_1')(x)
x = conv_prelu(64, (3, 3), 'conv2_2')(x)
x = layers.Dropout(0.25)(x)

pool2_output = layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)

conv3_1 = conv_prelu(64, (3, 3), 'conv3_1')(pool2_output)
conv3_2 = conv_prelu(64, (3, 3), 'conv3_2')(conv3_1)
pool3 = layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(conv3_2)
conv4_1 = conv_prelu(64, (3, 3), 'conv4_1')(pool3)
conv4_2 = conv_prelu(64, (3, 3), 'conv4_2')(conv4_1)

conv5_2 = conv_prelu(64, (3, 3), 'conv5_2')(pool2_output)
conv5_3 = conv_prelu(64, (3, 3), 'conv5_3')(conv5_2)

conv5_1 = conv_prelu(64, (3, 3), 'conv5_1')(pool2_output)
concat_1 = layers.Concatenate(name="concat_1")([conv3_2, conv5_1, conv5_3])
pool5 = layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool5')(concat_1)

conv6_2 = conv_prelu(64, (3, 3), 'conv6_2')(pool5)
conv6_3 = conv_prelu(64, (3, 3), 'conv6_3')(conv6_2)

conv6_1 = conv_prelu(64, (3, 3), 'conv6_1')(pool5)
concat_2 = layers.Concatenate(name="concat_2")([conv4_2, conv6_1, conv6_3])

pool4 = layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(concat_2)

flatten = Flatten()(pool4)

fc = layers.Dense(32, name='fc', kernel_regularizer=l2(0.01))(flatten)
fc = layers.Dense(128, name='fc_2', kernel_regularizer=l2(0.01))(fc)
fc = layers.Dropout(0.5)(fc)
fc_class = layers.Dense(64, name='fc_class', kernel_regularizer=l2(0.01))(fc)

outputs = layers.Dense(1, activation='sigmoid', name='out')(fc_class)

model2 = tf.keras.Model(inputs=inputs, outputs=outputs)
model2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

models = {
    model2: "modelo Net2 intentando reducir overfitting",
}

for model, description in models.items():
    metrics = TrainingMetrics(model, resultsPathServer, modelDescription=description)
    metrics.batches_train(folderPath=routeServer, nPerBatch=4, epochs=3, isSequence=False)
