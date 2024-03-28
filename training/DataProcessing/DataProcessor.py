import os
import random
from datetime import datetime
import pandas as pd
import cv2
from FaceReconModule import FaceExtractorMultithread

class DataProcessor:
    def __init__(self, baseDirectory, destinationDirectory, sampleDirectory = None, sampleProbability = 0.1):
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

        #comenzamos a procesar las imagenes y videos
        self.processImages()
        self.processVideos()

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
                cv2.imwrite(f'{self.sampleDirectory}/{label}_{datetime.now()}.jpg', img)

        # Si tenemos 10000 imagenes guardadas, creamos un dataset con ellas y las guardamos en un fichero h5, borrando las imagenes de la memoria
        if len(self.faces) == 10000:
            #guardamos el número de imagenes, reales y falsas
            self.totalFaces += len(self.faces)
            self.totalFake += sum(self.labels)
            self.totalReal += len(self.faces) - sum(self.labels)
            #guardamos los datos y hacemos reset de arrays
            df = pd.DataFrame({'face': self.faces, 'label': self.labels})
            if not os.path.exists(os.path.join(self.destinationDirectory, 'dataframes')):
                os.makedirs(os.path.join(self.destinationDirectory, 'dataframes'))
            df.to_hdf(f'{os.path.join(self.destinationDirectory, 'dataframes')}/dataframe_{self.currentDatasetCounter}.h5', key=f'df{self.currentDatasetCounter}', mode='w')
            self.faces = []
            self.labels = []
            self.currentDatasetCounter += 1

    def processImages(self):
        for index,path in enumerate(self.imagesPaths):
            # Procesamos la imagen y la guardamos como un vector de vectores rgb
            img = cv2.imread(path)
            # Descartamos la imagen si tiene tamaño 1x1 o si tiene tamaño 85x180 (tamaños observados de imágenes cargadas mal en el dataset)
            if img.shape[0] == 1 or img.shape[0] == 85:
                continue # La saltamos
            img = cv2.resize(img, (200, 200))
            self.storeImage(img, self.imagesLabels[index])

    def processVideos(self):    
        face_extractor = FaceExtractorMultithread(percentageExtractionFake=0.9, percentageExtractionReal=0.75)
    
        tmpFaces,tmpLabels = face_extractor.transform(self.videosPaths, self.videosLabels)
        for i in range(len(tmpFaces)):
            self.storeImage(tmpFaces[i], tmpLabels[i])

            
processor = DataProcessor(baseDirectory='P:\TFG\Datasets',
                          destinationDirectory='P:\TFG\Datasets\dataframes\\valid\dataframes_correct',
                          sampleDirectory='P:\TFG\Datasets\dataframes\\valid\samples')
        

      