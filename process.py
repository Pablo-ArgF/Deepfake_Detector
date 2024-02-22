import os
import pandas as pd
from faceRecon import FaceExtractorMultithread,FaceExtractorMultithread_DeepLearning, FaceExtractor
from sklearn.utils import shuffle



class VideoProcessor():
    def __init__(self,baseDir,nameDataset,destinationDir,numFragments,eachNframes):
        self.baseDir=baseDir
        self.nameDataset = nameDataset
        self.destinationDir= destinationDir
        self.numFragments = numFragments
        self.eachNframes = eachNframes

        self.videos = []
        self.labels = []
        self.currentIsReal = True

    def processFolder(self, path = None):
        if(path == None):
            path = self.baseDir
        #check if current folder indicates real/fake
        sections = path.split('-')
        if(sections[-1] == 'real' or sections[-1] == 'fake'):
            if sections[-1] == 'real':
                self.currentIsReal = True
            else:
                self.currentIsReal = False


        for item in os.listdir(path):
            currentPath = os.path.join(path,item)
            if not os.path.isdir(currentPath):
                self.processVideo(currentPath)
            else:
                self.processFolder(currentPath)

    def processVideo(self,path):
        #check it is an pm4 file, if not return
        fragments = path.split('.')
        if(fragments[-1] != 'mp4'):
            return
        self.videos.append(path)
        if(self.currentIsReal):
            self.labels.append(1) 
        else:
            self.labels.append(0)

    def getDataFrame(self):
        self.processFolder()
        df = pd.DataFrame({'video': self.videos, 'label': self.labels})
        df = shuffle(df)
        return df
    
    def processFaces(self):
        dataFrame = self.getDataFrame()

        #dataFrame = dataFrame.sample(10, random_state=42)
        print(f'Processing {len(dataFrame)} videos')
        
        face_extractor = FaceExtractorMultithread(n=self.eachNframes)
        print('Extracting faces from videos...')
        fragmentSize = int(len(dataFrame)/self.numFragments)
        for i in range(63, self.numFragments):
            print(f'Processing fragment {i+1}/{self.numFragments}')
            processed = face_extractor.transform(dataFrame.iloc[fragmentSize*i : fragmentSize*(i+1)])
            # Guardamos el fragmento procesado en un fichero hdf
            processed.to_hdf(f'{self.destinationDir}\dataframe{i}_{self.nameDataset}.h5', key=f'df{i}', mode='w')

    
processor = VideoProcessor('E:\TFG\Datasets\FaceForensics','FaceForensics','E:\TFG\Datasets\dataframes_test\FaceForensics',100,50 )

processor.processFaces()
