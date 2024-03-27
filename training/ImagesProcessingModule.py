import os
import pandas as pd
import cv2

class ImageDatasetProcessor:
    def __init__(self, base_folder, destination_dir):
        self.base_folder = base_folder
        self.destination_dir = destination_dir
        self.images = []
        self.current_dataset = 0

    def process_folder(self, path):
        for item in os.listdir(path):
            current_path = os.path.join(path, item)
            if not os.path.isdir(current_path):
                self.process_image(current_path)
            else:
                self.process_folder(current_path)

    def save_dataset(self):
        # Creamos un dataset con las imagenes en 'face' y en 'label' el valor 0 para todas
        # Guardamos el dataset en un fichero hdf
        dataset = pd.DataFrame({'face': self.images, 'label': [0]*len(self.images)})
        dataset.to_hdf(f'{self.destination_dir}/dataframe{self.current_dataset}_IMDB_Wiki.h5', key=f'df{self.current_dataset}', mode='w')
        self.current_dataset += 1

    def process_image(self, path):
        # Check if it is a jpg file, if not, return
        fragments = path.split('.')
        if fragments[-1] != 'jpg':
            return
        # Procesamos la imagen y la guardamos como un vector de vectores rgb
        img = cv2.imread(path)
        # Descartamos la imagen si tiene tamaño 1x1 o si tiene tamaño 85x180 (tamaños observados de imágenes cargadas mal en el dataset)
        if img.shape[0] == 1 or img.shape[0] == 85:
            return
        img = cv2.resize(img, (200, 200))
        self.images.append(img)
        # Cada vez que lleguemos a 10000 imágenes guardamos el dataset
        if len(self.images) >= 10000:
            self.save_dataset()
            self.images.clear()

    def process_dataset(self):
        self.process_folder(self.base_folder)

# Usage
baseFolder = 'P:\\TFG\\Datasets\\IMDB-Wiki face image dataset\\wiki_crop'
destinationDir = 'P:\\TFG\\Datasets\\dataframes\\valid\\dataframes_moreReal'

processor = ImageDatasetProcessor(baseFolder, destinationDir)
processor.process_dataset()
