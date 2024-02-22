import os
import pandas as pd
from faceRecon import FaceExtractorMultithread, FaceExtractor
from sklearn.utils import shuffle


baseDir='P:\TFG\Datasets\FaceForensics'
nameDataset = 'FaceForensics'
destinationDir='P:\TFG\Datasets\dataframes_test'
numFragments = 100
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
    #check it is an pm4 file, if not return
    fragments = path.split('.')
    if(fragments[-1] != 'mp4'):
        return
    videos.append(path)
    if(currentIsReal):
        labels.append(1) 
    else:
        labels.append(0)
    

dataFrame = processFolder(baseDir)
dataFrame = pd.DataFrame({'video': videos, 'label': labels})
# Mezclamos el dataframe para que no aparezcan las categorías seguidas
#dataFrame = shuffle(dataFrame)
print(len(dataFrame))

# Reduce el tamaño del dataset para que sea más fácil de manejar
# dataFrame = dataFrame.sample(10, random_state=42)

face_extractor = FaceExtractorMultithread(n=eachNframes)
print('Extracting faces from videos...')
fragmentSize = int(len(dataFrame)/numFragments)
for i in range(numFragments):
    print(f'Processing fragment {i+1}/{numFragments}')
    processed = face_extractor.transform(dataFrame.iloc[fragmentSize*i : fragmentSize*(i+1)])
    # Guardamos el fragmento procesado en un fichero hdf
    processed.to_hdf(f'{destinationDir}\dataframe{i}_{nameDataset}.h5', key=f'df{i}', mode='w')
