import os
import pandas as pd
from faceRecon import FaceExtractorMultithread, FaceExtractor


baseDir='E:\TFG\Datasets\FaceForensics'
nameDataset = 'FaceForensics'
destinationDir='E:\TFG\Datasets\dataframes'
numFragments = 10
eachNframes = 20

videos = []
labels = []
currentIsReal = True

def processFolder(path):
    #check if current folder indicates real/fake
    sections = path.split('-')
    if(sections[-1] == 'real' or sections[-1] == 'fake'):
        currentIsReal = sections[-1] == 'real'

    for item in os.listdir(path):
        currentPath = os.path.join(path,item)
        
        if not os.path.isdir(currentPath):
            processVideo(currentPath)
        else:
            processFolder(currentPath)

def processVideo(path):
    videos.append(path)
    if(currentIsReal):
        labels.append(1) 
    else:
        labels.append(0)
    

dataFrame = processFolder(baseDir)
dataFrame = pd.DataFrame({'video': videos, 'label': labels})
print(len(dataFrame))
dataFrame = dataFrame.sample(10, random_state=42)



        


# Reduce el tamaño del dataset para que sea más fácil de manejar
dataFrame = dataFrame.sample(10, random_state=42)

face_extractor = FaceExtractorMultithread(n=eachNframes)
print('Extracting faces from videos...')
fragmentSize = int(len(dataFrame)/numFragments)
for i in range(numFragments):
    print(f'Processing fragment {i+1}/{numFragments}')
    processed = face_extractor.transform(dataFrame.iloc[fragmentSize*i : fragmentSize*(i+1)])
    # Guardamos el fragmento procesado en un fichero hdf
    processed.to_hdf(f'{destinationDir}\dataframe{i}_{nameDataset}.h5', key=f'df{i}', mode='w')
