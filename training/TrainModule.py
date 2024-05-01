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
value_PReLU = 0.25
# PReLU(alpha_initializer=Constant(value=value_PReLU))   #TODO meter prelu y no poner activationi relu en todas /%)$(/%=)$(=%/=)·(/%=)(·/%=%)(·/%=)(%/·=)(%/·=

"""
# Create a Sequential model
model = Sequential()

# Input Layer
model.add(layers.InputLayer(input_shape=(200, 200, 3))) 

# Conv1_1 and Conv1_2 Layers
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_1'))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2'))

# Pool1 Layer
model.add(layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool1'))

# Conv2_1 and Conv2_2 Layers
model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_1'))
model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2'))

# Pool2 Layer
pool2 = layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool2')
model.add(pool2)

seq1 = Sequential()
seq3 = Sequential()
seq4 = Sequential()

#Configuring the seq1
seq1.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3_1')(pool2))
conv32 = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3_2')
seq1.add(conv32)
seq1.add(layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool3'))
seq1.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv4_1'))
seq1.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv4_2'))

#Configuring the seq3
seq3.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv5_2')(pool2))
seq3.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv5_3'))


#Configuration of the middle sequence
conv51 = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv5_1')(pool2.output)
model.add(layers.Concatenate(name="concat_1")([conv32.output, conv51.output,seq3.output]))
pool5 = layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool5')
model.add(pool5)

#Configuring the seq4
seq4.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv6_2')(pool5))
seq4.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv6_3'))

conv61 = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv6_1')(pool5.output)
model.add(layers.Concatenate(name="concat_2")([seq1.output, conv61.output ,seq4.output]))

model.add(layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool4'))

model.add(layers.Dense(units=1024, name='fc'))

model.add(layers.Dense(units=4096, name='fc_class'))

# Softmax Output Layer
model.add(layers.Softmax(units=1))

# Compile the model (add optimizer, loss function, etc.)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
"""

# Input Layer
inputs = layers.Input(shape=(200, 200, 3))

# Conv1_1 and Conv1_2 Layers
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_1')(inputs)
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2')(x)

# Pool1 Layer
x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)

# Conv2_1 and Conv2_2 Layers
x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_1')(x)
x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2')(x)

# Pool2 Layer
pool2_output = layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)

# Now you can use pool2_output as input for other layers
conv3_1 = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3_1')(pool2_output)
conv3_2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3_2')(conv3_1)
pool3 = layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(conv3_2)
conv4_1 = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv4_1')(pool3)
conv4_2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv4_2')(conv4_1)

conv5_2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv5_2')(pool2_output)
conv5_3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv5_3')(conv5_2)

conv5_1 = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv5_1')(pool2_output)
concat_1 = layers.Concatenate(name="concat_1")([conv3_2, conv5_1, conv5_3])
pool5 = layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool5')(concat_1)

conv6_2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv6_2')(pool5)
conv6_3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv6_3')(conv6_2)

conv6_1 = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv6_1')(pool5)
concat_2 = layers.Concatenate(name="concat_2")([conv4_2, conv6_1, conv6_3])

pool4 = layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(concat_2)

flatten = Flatten()(pool4)

fc = layers.Dense(1024, name='fc')(flatten)

fc_class = layers.Dense(4096, name='fc_class')(fc)

# Softmax Output Layer
outputs = layers.Dense(1, activation='sigmoid', name='out')(fc_class)

# Compile the model (add optimizer, loss function, etc.)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])



#---------------------------------------------------------------------------------------------------------------------------------------------

#entremamos secuencialmente los modelos
models = {
    model : "modelo Net2 del paper Facial Feature Extraction Method Based on Shallow and Deep Fusion CNN"
}

for model,description in models.items():
    metrics = TrainingMetrics(model, resultsPathServer, modelDescription = description)
    metrics.batches_train(routeServer,nBatches = 9 , epochs = 3) # Divide the hole dataset into <nbatches> fragments and train <epochs> epochs with each


