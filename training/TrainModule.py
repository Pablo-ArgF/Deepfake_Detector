import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import Dense, MaxPooling2D, Conv2D, Flatten,Lambda,  Input,Dropout, PReLU
from tensorflow.keras.models import Sequential
from MetricsModule import TrainingMetrics


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Para que no muestre los warnings de tensorflow

route = 'P:\TFG\Datasets\dataframes_small' #'/home/pabloarga/Data'
resultsPath = 'P:\TFG\Datasets\dataframes_small\\results' #'/home/pabloarga/Results2' 

routeServer = '/home/pabloarga/Data'
resultsPathServer = '/home/pabloarga/Results' 

"""
model2 = Sequential()
model2.add(Input(shape=(200, 200, 3))) 
model2.add(Conv2D(16, (3, 3), activation='relu'))
model2.add(MaxPooling2D((2, 2)))
model2.add(Conv2D(64, (3, 3), activation='relu'))
model2.add(MaxPooling2D((2, 2)))
model2.add(Conv2D(64, (3, 3), activation='relu'))
model2.add(Flatten())
model2.add(Dense(64, activation='relu'))
model2.add(Dense(1, activation='sigmoid'))
model2.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
"""

#---------------------------------------------------------------------------------------------------------------------------------------------
"""
value_PReLU = 0.25
model3 = Sequential([
    Input(shape=(200, 200, 3)),
    
    Conv2D(16, (3, 3), strides=(1, 1)),
    PReLU(alpha_initializer=Constant(value=value_PReLU)),
    Conv2D(16, (3, 3), strides=(1, 1)),
    PReLU(alpha_initializer=Constant(value=value_PReLU)),
    MaxPooling2D((2, 2), strides=(2, 2)),
    
    Conv2D(16, (3, 3), strides=(1, 1)),
    PReLU(alpha_initializer=Constant(value=value_PReLU)),
    Conv2D(16, (3, 3), strides=(1, 1)),
    PReLU(alpha_initializer=Constant(value=value_PReLU)),
    MaxPooling2D((2, 2), strides=(2, 2)),
    
    Conv2D(16, (3, 3), strides=(1, 1)),
    PReLU(alpha_initializer=Constant(value=value_PReLU)),
    Conv2D(16, (3, 3), strides=(1, 1)),
    PReLU(alpha_initializer=Constant(value=value_PReLU)),
    MaxPooling2D((2, 2), strides=(2, 2)),
    
    Conv2D(16, (3, 3), strides=(1, 1)),
    PReLU(alpha_initializer=Constant(value=value_PReLU)),
    Conv2D(16, (3, 3), strides=(1, 1)),
    PReLU(alpha_initializer=Constant(value=value_PReLU)),
    Conv2D(16, (1, 1), strides=(1, 1)),
    PReLU(alpha_initializer=Constant(value=value_PReLU)),
    Conv2D(16, (1, 1), strides=(1, 1)),
    PReLU(alpha_initializer=Constant(value=value_PReLU)),
    Conv2D(16, (3, 3), strides=(1, 1)),
    PReLU(alpha_initializer=Constant(value=value_PReLU)),
    
    MaxPooling2D((2, 2), strides=(2, 2)),
    
    Conv2D(16, (1, 1), strides=(1, 1)),
    PReLU(alpha_initializer=Constant(value=value_PReLU)),
    Conv2D(16, (1, 1), strides=(1, 1)),
    PReLU(alpha_initializer=Constant(value=value_PReLU)),
    Conv2D(16, (3, 3), strides=(1, 1)),
    PReLU(alpha_initializer=Constant(value=value_PReLU)),
    
    MaxPooling2D((2, 2), strides=(2, 2)),
    
    Flatten(),
    Dense(64, activation='softmax'),
    Dropout(0.25),
    Dense(1, activation='sigmoid')
])

model3.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model4 = Sequential([
    Input(shape=(200, 200, 3)),
    
    Conv2D(16, (5, 5), strides=(2, 2), padding='same'),
    PReLU(alpha_initializer=Constant(value=value_PReLU)),
    MaxPooling2D((2, 2), strides=(2, 2)),
    
    Conv2D(16, (5, 5), strides=(2, 2), padding='same'),
    PReLU(alpha_initializer=Constant(value=value_PReLU)),
    MaxPooling2D((2, 2), strides=(2, 2)),
    
    Conv2D(16, (5, 5), strides=(2, 2), padding='same'),
    PReLU(alpha_initializer=Constant(value=value_PReLU)),
    MaxPooling2D((2, 2), strides=(2, 2)),
    
    Conv2D(16, (5, 5), strides=(2, 2), padding='same'),
    PReLU(alpha_initializer=Constant(value=value_PReLU)),
    MaxPooling2D((2, 2), strides=(2, 2)),
    
    Flatten(),
    Dense(64, activation='softmax'),
    Dropout(0.25),
    Dense(1, activation='sigmoid')
])

model4.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

"""
value_PReLU = 0.25
model3 = Sequential([
    Input(shape=(200, 200, 3)),
    
    Conv2D(16, (3, 3), strides=(1, 1)),
    PReLU(alpha_initializer=Constant(value=value_PReLU)),
    Conv2D(16, (3, 3), strides=(1, 1)),
    PReLU(alpha_initializer=Constant(value=value_PReLU)),
    MaxPooling2D((2, 2), strides=(2, 2)),
    
    Conv2D(16, (3, 3), strides=(1, 1)),
    PReLU(alpha_initializer=Constant(value=value_PReLU)),
    Conv2D(16, (3, 3), strides=(1, 1)),
    PReLU(alpha_initializer=Constant(value=value_PReLU)),
    MaxPooling2D((2, 2), strides=(2, 2)),
    
    Conv2D(16, (3, 3), strides=(1, 1)),
    PReLU(alpha_initializer=Constant(value=value_PReLU)),
    Conv2D(16, (3, 3), strides=(1, 1)),
    PReLU(alpha_initializer=Constant(value=value_PReLU)),
    Conv2D(16, (1, 1), strides=(1, 1)),
    PReLU(alpha_initializer=Constant(value=value_PReLU)),
    Conv2D(16, (1, 1), strides=(1, 1)),
    PReLU(alpha_initializer=Constant(value=value_PReLU)),
    Conv2D(16, (3, 3), strides=(1, 1)),
    PReLU(alpha_initializer=Constant(value=value_PReLU)),
    
    MaxPooling2D((2, 2), strides=(2, 2)),
    
    Conv2D(16, (1, 1), strides=(1, 1)),
    PReLU(alpha_initializer=Constant(value=value_PReLU)),
    Conv2D(16, (1, 1), strides=(1, 1)),
    PReLU(alpha_initializer=Constant(value=value_PReLU)),
    Conv2D(16, (3, 3), strides=(1, 1)),
    PReLU(alpha_initializer=Constant(value=value_PReLU)),
    
    MaxPooling2D((2, 2), strides=(2, 2)),
    
    Flatten(),
    Dense(64, activation='softmax'),
    Dropout(0.25),
    Dense(1, activation='sigmoid')
])

model3.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model4 = Sequential([
    Input(shape=(200, 200, 3)),

    Conv2D(16, (5, 5), strides=(2, 2), padding='same'),
    PReLU(alpha_initializer=Constant(value=value_PReLU)),
    MaxPooling2D((2, 2), strides=(2, 2)),
    
    Conv2D(16, (5, 5), strides=(2, 2), padding='same'),
    PReLU(alpha_initializer=Constant(value=value_PReLU)),
    MaxPooling2D((2, 2), strides=(2, 2)),
    
    Conv2D(16, (5, 5), strides=(2, 2), padding='same'),
    PReLU(alpha_initializer=Constant(value=value_PReLU)),
    MaxPooling2D((2, 2), strides=(2, 2)),
    
    Flatten(),
    Dense(64, activation='softmax'),
    Dropout(0.25),
    Dense(1, activation='sigmoid')
])

model4.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
#---------------------------------------------------------------------------------------------------------------------------------------------

#entremamos secuencialmente los modelos
models = {
    model4 : "modelo reduced Net1 (modificado en las conv layers para que sea (2,2) en vez de (3,3)) del paper Facial Feature Extraction Method Based on Shallow and Deep Fusion CNN",
    model3 : "modelo reduced Net2 del paper Facial Feature Extraction Method Based on Shallow and Deep Fusion CNN"
    
}

for model,description in models.items():
    metrics = TrainingMetrics(model, resultsPathServer, modelDescription = description)
    metrics.batches_train(routeServer,nBatches = 41 , epochs = 3) # Divide the hole dataset into <nbatches> fragments and train <epochs> epochs with each


