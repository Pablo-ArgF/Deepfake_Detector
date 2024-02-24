import os
from tensorflow.keras.layers import Dense, MaxPooling2D, Conv2D, Flatten,Lambda,  Input,Dropout
from tensorflow.keras.models import Sequential
from MetricsModule import TrainingMetrics

route =  'P:\TFG\Datasets\dataframes_small' #'/home/pabloarga/Data' 
resultsPath = 'P:\TFG\Datasets\dataframes_small\\results' #'/home/pabloarga/Results'

model = Sequential()
model.add(Input(shape=(200, 200, 3)))
model.add(Lambda(lambda x: x/255.0)) #normalizamos los valores de los pixeles -> mejora la eficiencia
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dropout(0.2))  # Dropout for regularization
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))  # Dropout for regularization
model.add(Dense(1024, activation='relu'))
model.add(Dense(1, activation='softmax'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

metrics = TrainingMetrics(model, resultsPath)
metrics.batches_train(route,2,1)


#exportamos el modelo
model.save(os.path.join(resultsPath,'model.keras'))
