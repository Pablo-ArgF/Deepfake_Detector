import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, MaxPooling2D, Conv2D, Flatten,Lambda,  Input,Dropout
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
model2.add(Conv2D(32, (3, 3), activation='relu'))
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

model3 = Sequential()
model3.add(Input(shape=(200, 200, 3))) 
model3.add(Conv2D(32, (3, 3), activation='relu'))
model3.add(MaxPooling2D((2, 2)))
model3.add(Dropout(0.25))  # Add dropout after the first convolutional layer
model3.add(Conv2D(64, (3, 3), activation='relu'))
model3.add(MaxPooling2D((2, 2)))
model3.add(Dropout(0.25))  # Add dropout after the second convolutional layer
model3.add(Conv2D(64, (3, 3), activation='relu'))
model3.add(MaxPooling2D((2, 2)))
model3.add(Dropout(0.25))  # Add dropout after the third convolutional layer
model3.add(Flatten())
model3.add(Dense(64, activation='relu'))
model3.add(Dropout(0.5))  # Add dropout before the final dense layer
model3.add(Dense(32, activation='relu'))
model3.add(Dense(1, activation='sigmoid'))
model3.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])



base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(200, 200, 3))

# Congelar las capas del modelo base
for layer in base_model.layers:
    layer.trainable = False

# Crear un nuevo modelo encima del modelo base
model4 = tf.keras.models.Sequential()
model4.add(base_model)
model4.add(layers.Flatten())
model4.add(layers.Dense(256, activation='relu'))
model4.add(layers.Dense(1, activation='sigmoid'))

model4.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

#---------------------------------------------------------------------------------------------------------------------------------------------

#entremamos secuencialmente los modelos
models = {
    model3 : "Mejor estructura hasta la fecha + dropouts",
    model4 : "Modelo que utiliza VGG16 pretrained model como base y añade unas capas extra"
}

for model,description in models.items():
    metrics = TrainingMetrics(model, resultsPathServer, modelDescription = description)
    metrics.batches_train(routeServer,nBatches = 25 , epochs = 5) # Divide the hole dataset into <nbatches> fragments and train <epochs> epochs with each


