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

routeServer = '/home/pabloarga/Data/dataframes'
resultsPathServer = '/home/pabloarga/Results' 

#---------------------------------------------------------------------------------------------------------------------------------------------
# Function to add Convolutional layer with PReLU activation
def conv_prelu(filters, kernel_size, name):
    conv_layer = layers.Conv2D(filters, kernel_size, padding='same', name=name)
    prelu_layer = PReLU(alpha_initializer=Constant(value=value_PReLU))
    return Sequential([conv_layer, prelu_layer])
    
# Define the constant value for PReLU alpha
value_PReLU = 0.25

#---------------------------------------------------------------------------------------------------------------------------------------------
# Input Layer
inputs = layers.Input(shape=(200, 200, 3))

# Conv1_1 and Conv1_2 Layers
x = conv_prelu(32, (3, 3), 'conv1_1')(inputs)
x = conv_prelu(32, (3, 3), 'conv1_2')(x)
x = layers.Dropout(0.25)(x)  # Adding dropout after Conv1_2

# Pool1 Layer
x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)

# Conv2_1 and Conv2_2 Layers
x = conv_prelu(64, (3, 3), 'conv2_1')(x)
x = conv_prelu(64, (3, 3), 'conv2_2')(x)
x = layers.Dropout(0.25)(x)  # Adding dropout after Conv2_2

# Pool2 Layer
pool2_output = layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)

# Now you can use pool2_output as input for other layers
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

fc = layers.Dense(64, name='fc')(flatten)
fc = layers.Dropout(0.5)(fc)  # Adding dropout before the fully connected layer

fc_class = layers.Dense(128, name='fc_class')(fc)

# Softmax Output Layer
outputs = layers.Dense(1, activation='sigmoid', name='out')(fc_class)

# Compile the model (add optimizer, loss function, etc.)
model2 = tf.keras.Model(inputs=inputs, outputs=outputs)
model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


#---------------------------------------------------------------------------------------------------------------------------------------------

#entremamos secuencialmente los modelos
models = {
    model2 : "modelo Net2 (reducido) - con Prelus y con dropouts - con dataset grande y rotacion de caras - del paper Facial Feature Extraction Method Based on Shallow and Deep Fusion CNN",
}

for model,description in models.items():
    metrics = TrainingMetrics(model, resultsPathServer, modelDescription = description)
    metrics.batches_train(folderPath = routeServer,nPerBatch = 5 , epochs = 2, isSequence = False) # Divide the hole dataset into <nbatches> fragments and train <epochs> epochs with each


