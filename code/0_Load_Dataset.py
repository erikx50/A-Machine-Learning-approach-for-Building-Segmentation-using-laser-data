import os
from datasets import load_dataset
from PIL import Image
import numpy as np
from tqdm import tqdm


def load_mapai():
    """
    Downloads the MapAI dataset from huggingface and saves it to a dataset folder.
    """

    # Unpack images, labels and LiDAR and store them in data folder
    dataset = load_dataset('sjyhne/mapai_dataset')
    print(dataset)

    # Create dataset folder if it doesnt exist
    dataset_path = os.path.normpath('../dataset')
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    # Create MapAI folder if it doesnt exist
    dataset_path = os.path.normpath('../dataset/MapAI')
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    # Iterate through MapAI_dataset
    for set_name in tqdm(dataset.keys()):
        # Create subfolder if it doesnt exist
        subfolder_path = os.path.normpath('../dataset/MapAI/' + set_name)
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)

        # Create subsubfolders if they doesnt exit
        subsubfolders = ['image', 'mask', 'lidar']
        for folder_name in subsubfolders:
            subsubfolder_path = os.path.normpath('../dataset/MapAI/' + set_name + '/' + folder_name)
            if not os.path.exists(subsubfolder_path):
                os.makedirs(subsubfolder_path)

        # Access data
        image_files = dataset[set_name]['image']
        mask_files = dataset[set_name]['mask']
        lidar_files = dataset[set_name]['lidar']
        filename_files = dataset[set_name]['filename']

        for x in range(len(filename_files)):
            # Remove file extension from filename
            filename = filename_files[x].split('.')[0]

            # Save RGB aerial image in image folder
            image_files[x].save(os.path.normpath('../dataset/MapAI/' + set_name + '/image/' + filename + '.PNG'))

            # Save Greyscale mask image in mask folder
            mask_array = np.asarray(mask_files[x])
            mask_im = Image.fromarray(mask_array * 255)     # Multiply pixels by 255: [0, 1] -> [0, 255]
            mask_im.save(os.path.normpath('../dataset/MapAI/' + set_name + '/mask/' + filename + '.PNG'))

            # Save floating point LiDAR data in lidar folder as a numpy file
            lidar_array = np.asarray(lidar_files[x])
            np.save(os.path.normpath('../dataset/MapAI/' + set_name + '/lidar/' + filename), lidar_array)


if __name__ == "__main__":
    load_mapai()
