import os
import random
import math
from datetime import datetime
import pandas as pd
import cv2
import sys
from FaceReconModule import FaceExtractorMultithread

class DataProcessor:
    def __init__(self, baseDirectory, destinationDirectory,sequenceLengths = None, sampleDirectory = None, sampleProbability = 0.001):
        self.baseDirectory = baseDirectory
        self.destinationDirectory = destinationDirectory
        self.currentDirectoryFake = False
        self.currentDatasetCounter = 0
        self.currentSequenceDatasetCounter = 0
        self.sampleDirectory = sampleDirectory
        self.sampleProbability = sampleProbability
        self.sequenceLengths = sequenceLengths

        self.imagesPaths = []
        self.imagesLabels = []
        self.videosPaths = []
        self.videosLabels = []

        #self.face_extractor = FaceExtractorMultithread(percentageExtractionFake=0.09, percentageExtractionReal=0.02) #TODO
        self.face_extractor = FaceExtractorMultithread(percentageExtractionFake=0.7, percentageExtractionReal=0.7)

        #resulting data
        self.faces = []
        self.labels = []

        #For each size passed as parameter in sequenceLengths, we will store an array containing the current sequences not saved yet
        self.sequencesData = []

        self.totalFaces = 0
        self.totalFake = 0
        self.totalReal = 0

        self.processFolder(baseDirectory)
        #guardamos en un archivo 'filesFound.csv' los paths y labels de las imagenes y videos encontrados
        df = pd.DataFrame({'paths': self.imagesPaths + self.videosPaths, 'labels': self.imagesLabels + self.videosLabels})
        df.to_csv(f'{self.destinationDirectory}/filesFound.csv', index=False)
        return

        #añadimos las headers al Progress.csv 
        with open(os.path.join(self.destinationDirectory,'Progress.csv'), 'w') as f:
            f.write('datasetNumber, datasetImages, datasetFake, datasetReal , totalFaces, totalFake, totalReal\n')

        #comenzamos a procesar las imagenes y videos
        self.processVideos()
        self.processImages()
        #guardamos aquellos que no hayan sido guardados en bloques 
        self.saveDataset()
        for i in range(len(self.sequenceLengths)):
            self.saveSequences(i)


    def processFolder(self, path):
        #check if current folder indicates real/fake
        sections = path.split('-')
        if(sections[-1] == 'real' or sections[-1] == 'fake'):
            if sections[-1] == 'real':
                self.currentDirectoryFake = False
            else:
                self.currentDirectoryFake = True

        for item in os.listdir(path):
            current_path = os.path.join(path, item)
            if not os.path.isdir(current_path):
                self.processFile(current_path)
            else:
                self.processFolder(current_path)

    def processFile(self, path):
        #Registramos la imagen o video en la lista correspondiente
        fragments = path.split('.')
        if fragments[-1] == 'jpg' or fragments[-1] == 'png':
            self.imagesPaths.append(path)
            if self.currentDirectoryFake:
                self.imagesLabels.append(1)
            else:
                self.imagesLabels.append(0)
        elif fragments[-1] == 'mp4':
            self.videosPaths.append(path)
            if self.currentDirectoryFake:
                self.videosLabels.append(1)
            else:
                self.videosLabels.append(0)

    def storeImage(self,img,label):
        # Guardamos la imagen en los arrays
        self.faces.append(img)
        self.labels.append(label)

        # Si hay sample directory y prob, guardamos la imagen en la carpeta sample con probabilidad sampleProbability
        if self.sampleDirectory:
            if random.random() < self.sampleProbability:
                if not os.path.exists(self.sampleDirectory):
                    os.makedirs(self.sampleDirectory)
                cv2.imwrite(f'{self.sampleDirectory}/{label}_image_{len(self.faces) + self.totalFaces}.jpg', img)

        # Si tenemos x imagenes guardadas, creamos un dataset con ellas y las guardamos en un fichero h5, borrando las imagenes de la memoria
        if len(self.faces) == 25000:
           self.saveDataset()

    def saveDataset(self):
        #guardamos el número de imagenes, reales y falsas
        self.totalFaces += len(self.faces)
        self.totalFake += sum(self.labels)
        self.totalReal += len(self.faces) - sum(self.labels)

        #escribimos en el archivo Progress.csv los datos de progreso actuales
        with open(os.path.join(self.destinationDirectory,'Progress.csv'), 'a') as f:
            f.write(f'{self.currentDatasetCounter},')
            f.write(f'{len(self.faces)},')
            f.write(f'{sum(self.labels)},')
            f.write(f'{len(self.faces) - sum(self.labels)},')
            f.write(f'{self.totalFaces},')
            f.write(f'{self.totalFake},')
            f.write(f'{self.totalReal}\n')

        #guardamos los datos y hacemos reset de arrays
        df = pd.DataFrame({'face': self.faces, 'label': self.labels})
        dataframeFolder = os.path.join(self.destinationDirectory, 'dataframes')
        if not os.path.exists(dataframeFolder):
            os.makedirs(dataframeFolder)
        df.to_hdf(f'{dataframeFolder}/dataframe_{self.currentDatasetCounter}.h5', key=f'df{self.currentDatasetCounter}', mode='w')
        self.faces = []
        self.labels = []
        self.currentDatasetCounter += 1

        


    def processImages(self):
        for index,path in enumerate(self.imagesPaths):
            if index % 100 == 0:
                print(f'image {index}/{len(self.imagesPaths)}')
            tmpFaces, tmpLabels = self.face_extractor.process_image(path, self.imagesLabels[index])
            for i in range(len(tmpFaces)):
                self.storeImage(tmpFaces[i], tmpLabels[i])


    def processVideos(self):    
        #dividimos la cantidad de videos de forma que se procesen de 100 en 100
        numberOfChunks = math.ceil(len(self.videosPaths) / 100) 
        for chunk in range(numberOfChunks):
            print(f'###################################### chunck {chunk}/{numberOfChunks} ################################')
            pathsChunk = self.videosPaths[chunk*100 : min(chunk*100 + 100, len(self.videosPaths))] 
            labelsChunk = self.videosLabels[chunk*100 : min(chunk*100 + 100, len(self.videosPaths))] 
            tmpFaces,tmpLabels = self.face_extractor.transform(pathsChunk, labelsChunk)
            for i in range(len(tmpFaces)):
                self.storeImage(tmpFaces[i], tmpLabels[i])    
            #Si tenemos sequenceLengths, generamos secuencias de longitud sequenceLengths 
            if self.sequenceLengths:
                self.registerSequences(tmpFaces,tmpLabels[0])  

    """
    Recibe todas las imagenes y label de un video y guarda las secuencias con los tamaños especificados por constructor.
    Por cada tamaño especificado en el constructor se creará una carpeta en la destination directory con el nombre: 'sequences_{sequenceLength}'
    """
    def registerSequences(self,faces,label):
        for index,sequenceLength in enumerate(self.sequenceLengths):
            sequences = [faces[i:i+sequenceLength] for i in range(0, len(faces), sequenceLength)] #TODO las que se salen las estoy descartando
            #store in the sequencesData array the sequences that are not saved yet
            self.sequencesData[index].extend([sequences , label])
            if len(self.sequencesData[index]) * self.sequenceLengths[index] >= 10000: #TODO modificar esto para que sea más, ahora lo puse así para los tests
                self.saveSequences(index)

    """
    Guarda las secuencias almacenadas en el array sequencesData en un fichero h5
    """
    def saveSequences(self,index):
        #guardamos los datos y hacemos reset de arrays
        df = pd.DataFrame({'sequences': self.sequencesData[index][0], 'label': self.sequencesData[index][1]})
        dataframeFolder = os.path.join(self.destinationDirectory, f'sequences_{self.sequenceLengths[index]}')
        if not os.path.exists(dataframeFolder):
            os.makedirs(dataframeFolder)

        #In order to test correctness we save all images in a sequence inside a folder in the dataframe folder
        for i,pair in enumerate(self.sequencesData[index]):
            sequence = pair[0]
            label = pair[1]
            sequenceFolder = os.path.join(dataframeFolder, f'sequence_{i}')
            if not os.path.exists(sequenceFolder):
                os.makedirs(sequenceFolder)
            for j,image in enumerate(sequence):
                cv2.imwrite(f'{sequenceFolder}/image_{j}_{label}.jpg', image)

        df.to_hdf(f'{dataframeFolder}/sequences_{self.currentSequenceDatasetCounter}.h5', key=f'df{self.currentSequenceDatasetCounter}', mode='w')
        self.sequencesData[index] = []
        self.currentSequenceDatasetCounter += 1
            
            
            

"""          
processor = DataProcessor(baseDirectory='E:\TFG\Datasets',
                          destinationDirectory='E:\TFG\Datasets\dataframes\\valid\dataframes_correct',
                          sampleDirectory='E:\TFG\Datasets\dataframes\\valid\samples')
"""
processor = DataProcessor(baseDirectory='P:\\TFG\\Data',
                          destinationDirectory='P:\\TFG\\Processed_Data\\valid\\dataframes',
                          sampleDirectory='P:\\TFG\\Processed_Data\\valid\\samples',
                          sequenceLengths=[20,50,100,300])        

      