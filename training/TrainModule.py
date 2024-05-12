import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import Dense, MaxPooling2D, Conv2D, Flatten,Lambda ,Input ,Dropout, PReLU
from tensorflow.keras.models import Sequential
from MetricsModule import TrainingMetrics


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Para que no muestre los warnings de tensorflow

route = 'P:\TFG\Datasets\dataframes_small' #'/home/pabloarga/Data'
resultsPath = 'P:\TFG\Datasets\dataframes_small\\results' #'/home/pabloarga/Results2' 

routeServer = '/home/pabloarga/Data'
resultsPathServer = '/home/pabloarga/Results' 

#---------------------------------------------------------------------------------------------------------------------------------------------
# Function to add Convolutional layer with PReLU activation
def conv_prelu(filters, kernel_size, name):
    return layers.Conv2D(filters, kernel_size, padding='same', name=name,
                              activation = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None))


inputs = layers.Input(shape=(200, 200, 3))

#Normalization layer
x = Lambda(lambda x: x/255.0)(inputs)

x = conv_prelu(64, (3, 3), 'conv1_1')(x)
x = conv_prelu(64, (3, 3), 'conv1_2')(x)

x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)

x = conv_prelu(128, (3, 3), 'conv2_1')(x)
x = conv_prelu(128, (3, 3), 'conv2_2')(x)

pool2_output = layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)

conv3_1 = conv_prelu(128, (3, 3), 'conv3_1')(pool2_output)
conv3_2 = conv_prelu(128, (3, 3), 'conv3_2')(conv3_1)
pool3 = layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(conv3_2)
conv4_1 = conv_prelu(128, (3, 3), 'conv4_1')(pool3)
conv4_2 = conv_prelu(128, (3, 3), 'conv4_2')(conv4_1)

conv5_2 = conv_prelu(128, (1, 1), 'conv5_2')(pool2_output)
conv5_3 = conv_prelu(128, (3, 3), 'conv5_3')(conv5_2)

conv5_1 = conv_prelu(128, (1, 1), 'conv5_1')(pool2_output)
concat_1 = layers.Concatenate(name="concat_1")([conv3_2, conv5_1, conv5_3])
pool5 = layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool5')(concat_1)

conv6_2 = conv_prelu(128, (1, 1), 'conv6_2')(pool5)
conv6_3 = conv_prelu(128, (3, 3), 'conv6_3')(conv6_2)

conv6_1 = conv_prelu(128, (1, 1), 'conv6_1')(pool5)
concat_2 = layers.Concatenate(name="concat_2")([conv4_2, conv6_1, conv6_3])

pool4 = layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(concat_2)

flatten = Flatten()(pool4)

fc = layers.Dense(1024, name='fc')(flatten)
fc = layers.Dropout(0.5)(fc)

fc_class = layers.Dense(2048, name='fc_class')(fc)
fc_class = layers.Dropout(0.5)(fc_class) 

outputs = layers.Dense(1, activation='softmax', name='out')(fc_class)

model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


#---------------------------------------------------------------------------------------------------------------------------------------------
inputs = layers.Input(shape=(200, 200, 3))

#Normalization layer
x = Lambda(lambda x: x/255.0)(inputs)

x = conv_prelu(64, (3, 3), 'conv1_1')(x)
x = conv_prelu(64, (3, 3), 'conv1_2')(x)

x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)

x = conv_prelu(128, (3, 3), 'conv2_1')(x)
x = conv_prelu(128, (3, 3), 'conv2_2')(x)

pool2_output = layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)

conv3_1 = conv_prelu(128, (3, 3), 'conv3_1')(pool2_output)
conv3_2 = conv_prelu(128, (3, 3), 'conv3_2')(conv3_1)
pool3 = layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(conv3_2)
conv4_1 = conv_prelu(128, (3, 3), 'conv4_1')(pool3)
conv4_2 = conv_prelu(128, (3, 3), 'conv4_2')(conv4_1)

conv5_2 = conv_prelu(128, (1, 1), 'conv5_2')(pool2_output)
conv5_3 = conv_prelu(128, (3, 3), 'conv5_3')(conv5_2)

conv5_4 = conv_prelu(128, (1, 1), 'conv5_4')(pool2_output)
conv5_5 = conv_prelu(128, (1, 1), 'conv5_5')(conv5_4)
conv5_6 = conv_prelu(128, (3, 3), 'conv5_6')(conv5_5)

conv5_1 = conv_prelu(128, (1, 1), 'conv5_1')(pool2_output)
concat_1 = layers.Concatenate(name="concat_1")([conv3_2, conv5_1, conv5_3, conv5_6])
pool5 = layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool5')(concat_1)

conv6_2 = conv_prelu(128, (1, 1), 'conv6_2')(pool5)
conv6_3 = conv_prelu(128, (3, 3), 'conv6_3')(conv6_2)

conv6_4 = conv_prelu(128, (1, 1), 'conv6_4')(pool5)
conv6_5 = conv_prelu(128, (1, 1), 'conv6_5')(conv6_4)
conv6_6 = conv_prelu(128, (3, 3), 'conv6_6')(conv6_5)

conv6_1 = conv_prelu(128, (1, 1), 'conv6_1')(pool5)
concat_2 = layers.Concatenate(name="concat_2")([conv4_2, conv6_1, conv6_3,conv6_6])

pool4 = layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(concat_2)

flatten = Flatten()(pool4)

fc = layers.Dense(1024, name='fc')(flatten)
fc = layers.Dropout(0.5)(fc)

fc_class = layers.Dense(2048, name='fc_class')(fc)
fc_class = layers.Dropout(0.5)(fc_class) 

outputs = layers.Dense(1, activation='softmax', name='out')(fc_class)

model2 = tf.keras.Model(inputs=inputs, outputs=outputs)
model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


#---------------------------------------------------------------------------------------------------------------------------------------------

#entremamos secuencialmente los modelos
models = {
    model : "WITH LAMBDA - using 1x1 convolutions in shallow features - modelo Net2 - softmax - con Prelus y con dropouts del paper Facial Feature Extraction Method Based on Shallow and Deep Fusion CNN",
    model2 : "net2 con un tercer livel de shallow features - con lambda - salida softmax"
}

for model,description in models.items():
    metrics = TrainingMetrics(model, resultsPathServer, modelDescription = description)
    metrics.batches_train(routeServer,nBatches = 9 , epochs = 2) # Divide the hole dataset into <nbatches> fragments and train <epochs> epochs with each


