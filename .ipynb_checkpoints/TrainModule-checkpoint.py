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

#---------------------------------------------------------------------------------------------------------------------------------------------

model = Sequential()
model.add(Input(shape=(200, 200, 3)))
model.add(Lambda(lambda x: x/255.0)) #normalizamos los valores de los pixeles -> mejora la eficiencia
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dropout(0.2))  # Dropout for regularization
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))  # Dropout for regularization
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))  # Dropout for regularization
model.add(Dense(1024, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


#---------------------------------------------------------------------------------------------------------------------------------------------

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


#---------------------------------------------------------------------------------------------------------------------------------------------

base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(200, 200, 3))

# Congelar las capas del modelo base
for layer in base_model.layers:
    layer.trainable = False

# Crear un nuevo modelo encima del modelo base
model3 = tf.keras.models.Sequential()
model3.add(base_model)
model3.add(layers.Flatten())
model3.add(layers.Dense(256, activation='relu'))
model3.add(layers.Dense(1, activation='sigmoid'))

model3.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

#---------------------------------------------------------------------------------------------------------------------------------------------

#entremamos secuencialmente los modelos
#models = [model, model2, model3]
models = [model2]

for model in models:
    metrics = TrainingMetrics(model, resultsPathServer)
    metrics.batches_train(routeServer,nBatches = 10 , epochs = 10) # Divide the hole dataset into <nbatches> fragments and train <epochs> epochs with each


