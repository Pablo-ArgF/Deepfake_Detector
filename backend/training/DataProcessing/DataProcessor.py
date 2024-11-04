import os
import random
import math
from datetime import datetime
import pandas as pd
import cv2
import sys
from FaceReconModule import FaceExtractorMultithread

class DataProcessor:
    """
    Clase para procesar datos extrayendo caras de imágenes y videos.

    :param baseDirectory: El directorio base que contiene los datos.
    :type baseDirectory: str
    :param destinationDirectory: El directorio de destino para guardar los datos procesados.
    :type destinationDirectory: str
    :param sequenceLengths: Lista de longitudes de secuencia para crear secuencias de fotogramas a partir de videos. Por defecto, None.
    :type sequenceLengths: list, opcional
    :param sampleDirectory: El directorio para guardar una muestra de imágenes procesadas. Por defecto, None.
    :type sampleDirectory: str, opcional
    :param sampleProbability: La probabilidad de guardar una imagen en el directorio de muestra. Por defecto, 0.001.
    :type sampleProbability: float, opcional

    :ivar baseDirectory: El directorio base que contiene los datos.
    :vartype baseDirectory: str
    :ivar destinationDirectory: El directorio de destino para guardar los datos procesados.
    :vartype destinationDirectory: str
    :ivar currentDirectoryFake: Indica si el directorio actual es para datos falsos.
    :vartype currentDirectoryFake: bool
    :ivar currentDatasetCounter: Contador para el conjunto de datos actual que se está procesando.
    :vartype currentDatasetCounter: int
    :ivar currentSequenceDatasetCounter: Contador para el conjunto de datos de secuencia actual que se está procesando.
    :vartype currentSequenceDatasetCounter: int
    :ivar sampleDirectory: El directorio para guardar una muestra de imágenes procesadas.
    :vartype sampleDirectory: str
    :ivar sampleProbability: La probabilidad de guardar una imagen en el directorio de muestra.
    :vartype sampleProbability: float
    :ivar sequenceLengths: Lista de longitudes de secuencia para crear secuencias de fotogramas a partir de videos.
    :vartype sequenceLengths: list
    :ivar imagesPaths: Lista de rutas de archivos de imágenes.
    :vartype imagesPaths: list
    :ivar imagesLabels: Lista de etiquetas para los archivos de imágenes.
    :vartype imagesLabels: list
    :ivar videosPaths: Lista de rutas de archivos de videos.
    :vartype videosPaths: list
    :ivar videosLabels: Lista de etiquetas para los archivos de videos.
    :vartype videosLabels: list
    :ivar face_extractor: Instancia de la clase FaceExtractorMultithread para extraer caras de imágenes y videos.
    :vartype face_extractor: FaceExtractorMultithread
    :ivar faces: Lista de caras extraídas.
    :vartype faces: list
    :ivar labels: Lista de etiquetas para las caras extraídas.
    :vartype labels: list
    :ivar sequencesData: Lista de matrices que contienen las secuencias actuales que aún no se han guardado para cada tamaño en sequenceLengths.
    :vartype sequencesData: list
    :ivar totalFaces: Número total de caras procesadas.
    :vartype totalFaces: int
    :ivar totalFake: Número total de caras falsas procesadas.
    :vartype totalFake: int
    :ivar totalReal: Número total de caras reales procesadas.
    :vartype totalReal: int

    :method processFolder: Procesa de forma recursiva una carpeta y sus subcarpetas.
    :method processFile: Procesa un archivo y lo registra en la lista correspondiente.
    :method storeImage: Almacena una imagen en las matrices y opcionalmente la guarda en el directorio de muestra.
    :method saveDataset: Guarda las caras y etiquetas almacenadas como un conjunto de datos en un archivo HDF5.
    :method processImages: Procesa los archivos de imágenes y extrae caras.
    :method processVideos: Procesa los archivos de videos y extrae caras.
    :method registerSequences: Registra secuencias de fotogramas de videos y las guarda como conjuntos de datos en archivos HDF5.
    :method saveSequences: Guarda las secuencias registradas como conjuntos de datos en archivos HDF5.
    """

    def __init__(self, baseDirectory, destinationDirectory,sequenceLengths = None, sampleDirectory = None, sampleProbability = 0.001):
        """
        Inicializa una instancia de la clase DataProcessor.

        Args:
            baseDirectory (str): El directorio base que contiene los datos.
            destinationDirectory (str): El directorio de destino para guardar los datos procesados.
            sequenceLengths (list, opcional): Lista de longitudes de secuencia para crear secuencias de fotogramas a partir de videos. Por defecto, None.
            sampleDirectory (str, opcional): El directorio para guardar una muestra de imágenes procesadas. Por defecto, None.
            sampleProbability (float, opcional): La probabilidad de guardar una imagen en el directorio de muestra. Por defecto, 0.001.
        """
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
        self.face_extractor = FaceExtractorMultithread(percentageExtractionFake=0.5, percentageExtractionReal=0.5)

        #resulting data
        self.faces = []
        self.labels = []

        #For each size passed as parameter in sequenceLengths, we will store an array containing the current sequences not saved yet
        self.sequencesData = [[] for i in range(len(sequenceLengths))]

        self.totalFaces = 0
        self.totalFake = 0
        self.totalReal = 0

        self.processFolder(baseDirectory)
        #guardamos en un archivo 'filesFound.csv' los paths y labels de las imagenes y videos encontrados
        df = pd.DataFrame({'paths': self.imagesPaths + self.videosPaths, 'labels': self.imagesLabels + self.videosLabels})
        df.to_csv(f'{self.destinationDirectory}/filesFound.csv', index=False)
        

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
        """
        Procesa una carpeta y todos sus archivos y subcarpetas.

        :param path: Ruta de la carpeta a procesar.
        """
        
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
        """
        Procesa un archivo y registra la imagen o video en la lista correspondiente.

        :param path: Ruta del archivo a procesar.
        :type path: str
        """
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
        """
            Almacena una imagen y su etiqueta en los arrays correspondientes.

            :param img: La imagen a almacenar.
            :type img: numpy.ndarray
            :param label: La etiqueta de la imagen.
            :type label: str
        """
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
        if len(self.faces) == 9000: 
           self.saveDataset()

    def saveDataset(self):
        """
            Guarda el dataset actual en disco y actualiza los datos de progreso.

            Este método guarda el dataset actual en disco en formato HDF5 y actualiza los datos de progreso en un archivo CSV.
            Los datos de progreso incluyen el número de imágenes totales, el número de imágenes falsas y el número de imágenes reales.

            Returns:
                None
        """        
        if len(self.faces) == 0:
            return
        print(f'--> saved dataset {self.currentDatasetCounter}')
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
        """
            Procesa las imágenes almacenadas en la lista de rutas de imágenes.
            Extrae las caras de las imágenes y las almacena junto con sus etiquetas en el almacenamiento.
        """
        for index,path in enumerate(self.imagesPaths):
            if index % 100 == 0:
                print(f'image {index}/{len(self.imagesPaths)}')
            tmpFaces, tmpLabels = self.face_extractor.process_image(path, self.imagesLabels[index])
            for i in range(len(tmpFaces)):
                self.storeImage(tmpFaces[i], tmpLabels[i])


    def processVideos(self):    
        """
            Procesa los videos divididos en chunks y realiza las siguientes tareas:
            - Divide la cantidad de videos en chunks de 100 videos cada uno.
            - Transforma los videos en secuencias de caras y etiquetas utilizando el extractor de caras.
            - Almacena las imágenes individualmente en dataframes para su uso en una red neuronal convolucional (CNN).
            - Registra las secuencias en dataframes para su uso en una red neuronal recurrente (RNN).

            Args:
                self: La instancia del objeto DataProcessor.

            Returns:
                None
        """        
        #dividimos la cantidad de videos de forma que se procesen de 100 en 100
        numberOfChunks = math.ceil(len(self.videosPaths) / 100) 
        for chunk in range(numberOfChunks):
            print(f'###################################### chunck {chunk+1}/{numberOfChunks} ################################')
            pathsChunk = self.videosPaths[chunk*100 : min(chunk*100 + 100, len(self.videosPaths))] 
            labelsChunk = self.videosLabels[chunk*100 : min(chunk*100 + 100, len(self.videosPaths))] 
            #Transform devuelve un array de arrays, cada array interno contiene las secuencias para un video, tambien devuelve un vector de labels para cada video
            videosFaces,videoLabels = self.face_extractor.transform(pathsChunk, labelsChunk)
            for index,video in enumerate(videosFaces):
                labels = videoLabels[index]
                #Guardado de imagenes individualmente en dataframes para CNN
                for j, face in enumerate(video):
                    self.storeImage(face,labels[j]) 
                #Guardado de secuencias en dataframes para RNN
                if self.sequenceLengths:
                    self.registerSequences(video,labels[0]) #Labels[0] porque todos los frames de un video tienen la misma label

    def registerSequences(self, faces, label):
        """
        Recibe todas las imagenes y label de un video y guarda las secuencias con los tamaños especificados por constructor.
        Por cada tamaño especificado en el constructor se creará una carpeta en la destination directory con el nombre: 'sequences_{sequenceLength}'

        :param faces: Lista de caras.
        :type faces: list
        :param label: Etiqueta asociada a las secuencias de caras.
        :type label: str
        """
        for index, sequenceLength in enumerate(self.sequenceLengths):
            sequences = [faces[i:i+sequenceLength] for i in range(0, len(faces), sequenceLength)] 
            # Almacena en el arreglo sequencesData las secuencias que aún no han sido guardadas
            for sequence in sequences:
                self.sequencesData[index].append([sequence, label]) 
            if len(self.sequencesData[index]) * self.sequenceLengths[index] >= 300: # TODO modificar esto para que sea más, ahora lo puse así para los tests
                self.saveSequences(index)

    def saveSequences(self,index):
        """
            Guarda las secuencias de datos en un archivo HDF5 y las imágenes en carpetas correspondientes.

            Parámetros:
            - index (int): El índice de la lista de secuencias de datos a guardar.

            Retorna:
            - None
        """
        #Si no hay nada que guardar, salimos
        if len(self.sequencesData[index]) == 0:
            return
        #guardamos los datos y hacemos reset de arrays
        sequences = []
        labels = []
        #Filtrado de secuencias que no tengan el tamaño correcto
        for pair in self.sequencesData[index]:
            if len(pair[0]) == self.sequenceLengths[index]:
                sequences.append(pair[0])
                labels.append(pair[1])
        #Si no hay nada que guardar, salimos
        if len(sequences) == 0:
            return
        df = pd.DataFrame({'sequences': sequences, 'label': labels})
        
        dataframeFolder = os.path.join(self.destinationDirectory, f'sequences_{self.sequenceLengths[index]}')
        if not os.path.exists(dataframeFolder):
            os.makedirs(dataframeFolder)

        #In order to test correctness we save all images in a sequence inside a folder in the dataframe folder
        
        for i,sequence in enumerate(sequences):
            sequenceFolder = os.path.join(dataframeFolder, f'sequence_{i}')
            if not os.path.exists(sequenceFolder):
                os.makedirs(sequenceFolder)
            for j,image in enumerate(sequence):
                cv2.imwrite(f'{sequenceFolder}/image_{j}.jpg', image)
        

        df.to_hdf(f'{dataframeFolder}/sequences_{self.currentSequenceDatasetCounter}.h5', key=f'df{self.currentSequenceDatasetCounter}', mode='w')
        self.sequencesData[index] = []
        self.currentSequenceDatasetCounter += 1
            
"""          
#ejemplo de procesado de datos sin generacion de secuencias
processor = DataProcessor(baseDirectory='E:\TFG\Datasets',
                            destinationDirectory='E:\TFG\Datasets\dataframes\\valid\dataframes_correct',
                            sampleDirectory='E:\TFG\Datasets\dataframes\\valid\samples')
                            
#ejemplo de procesado de datos con generacion de secuencias
processor = DataProcessor(baseDirectory='C:\\Users\\pablo\\Desktop\\TFG (1)\\videoFolder',
                            destinationDirectory='C:\\Users\\pablo\\Desktop\\TFG (1)\\test_datasets',
                            sampleDirectory='C:\\Users\\pablo\\Desktop\\TFG (1)\\samples',
                            sequenceLengths=[20,50,100,300])        
"""         
            


      