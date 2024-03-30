import os
import random
import math
from datetime import datetime
import pandas as pd
import cv2
from FaceReconModule import FaceExtractorMultithread

class DataProcessor:
    def __init__(self, baseDirectory, destinationDirectory, sampleDirectory = None, sampleProbability = 0.001):
        self.baseDirectory = baseDirectory
        self.destinationDirectory = destinationDirectory
        self.currentDirectoryFake = False
        self.currentDatasetCounter = 0
        self.sampleDirectory = sampleDirectory
        self.sampleProbability = sampleProbability

        self.imagesPaths = []
        self.imagesLabels = []
        self.videosPaths = []
        self.videosLabels = []

        self.face_extractor = FaceExtractorMultithread(percentageExtractionFake=0.09, percentageExtractionReal=0.02)

        #resulting data
        self.faces = []
        self.labels = []

        self.totalFaces = 0
        self.totalFake = 0
        self.totalReal = 0

        self.processFolder(baseDirectory)
        #guardamos en un archivo 'filesFound.csv' los paths y labels de las imagenes y videos encontrados
        df = pd.DataFrame({'paths': self.imagesPaths + self.videosPaths, 'labels': self.imagesLabels + self.videosLabels})
        df.to_csv(f'{self.destinationDirectory}/filesFound.csv', index=False)

        #añadimos las headers al Progress.csv 
        with open(os.path.join(self.destinationDirectory,'Progress.csv'), 'w') as f:
            f.write('datasetNumber, totalFaces, totalFake, totalReal\n')

        #comenzamos a procesar las imagenes y videos
        self.processVideos()
        self.processImages()
        #guardamos aquellos que no hayan sido guardados en bloques de 5000
        self.saveDataset()


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
        if fragments[-1] == 'jpg':
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
            if not os.path.exists(self.sampleDirectory):
                os.makedirs(self.sampleDirectory)
            if random.random() < self.sampleProbability:
                cv2.imwrite(f'{self.sampleDirectory}/{label}_image_{len(self.faces) + self.totalFaces}.jpg', img)

        # Si tenemos x imagenes guardadas, creamos un dataset con ellas y las guardamos en un fichero h5, borrando las imagenes de la memoria
        if len(self.faces) == 10000:
           self.saveDataset()

    def saveDataset(self):
        #guardamos el número de imagenes, reales y falsas
        self.totalFaces += len(self.faces)
        self.totalFake += sum(self.labels)
        self.totalReal += len(self.faces) - sum(self.labels)
        #guardamos los datos y hacemos reset de arrays
        df = pd.DataFrame({'face': self.faces, 'label': self.labels})
        dataframeFolder = os.path.join(self.destinationDirectory, 'dataframes')
        if not os.path.exists(dataframeFolder):
            os.makedirs(dataframeFolder)
        df.to_hdf(f'{dataframeFolder}/dataframe_{self.currentDatasetCounter}.h5', key=f'df{self.currentDatasetCounter}', mode='w')
        self.faces = []
        self.labels = []
        self.currentDatasetCounter += 1

        #escribimos en el archivo Progress.csv los datos de progreso actuales
        with open(os.path.join(self.destinationDirectory,'Progress.csv'), 'w') as f:
            f.write(f'{self.currentDatasetCounter -1 },')
            f.write(f'{self.totalFaces},')
            f.write(f'{self.totalFake},')
            f.write(f'{self.totalReal}\n')


    def processImages(self):
        for index,path in enumerate(self.imagesPaths):
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


            
processor = DataProcessor(baseDirectory='E:\TFG\Datasets',
                          destinationDirectory='E:\TFG\Datasets\dataframes\\valid\dataframes_correct',
                          sampleDirectory='E:\TFG\Datasets\dataframes\\valid\samples')
        

      