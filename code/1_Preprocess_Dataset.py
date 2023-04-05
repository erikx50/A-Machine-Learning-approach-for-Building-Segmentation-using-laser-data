import os
import cv2 as cv
from tqdm import tqdm
from eval_functions import _mask_to_boundary
import numpy as np
from tifffile import imwrite


def preprocess():
    """
    Preprocesses the MapAI dataset and saves the results to new folders.
    """

    # Defining sets that has to be rescaled
    datasets = ['train', 'validation', 'task1_test', 'task2_test']

    # Define image size
    new_size = 512

    for dataset in tqdm(datasets):
        # Make directory for preprocessed dataset
        dataset_path = os.path.normpath('../dataset/MapAI/preprocessed_' + dataset)
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)

        # Folder structure for data generator
        if dataset == 'train':
            label = 'train'
        elif dataset == 'validation':
            label = 'val'
        else:
            label = ''

        # Make directory for preprocessed subsets
        mask_path = os.path.normpath('../dataset/MapAI/preprocessed_' + dataset + '/mask/' + label)
        image_path = os.path.normpath('../dataset/MapAI/preprocessed_' + dataset + '/image/' + label)
        rgblidar_path = os.path.normpath('../dataset/MapAI/preprocessed_' + dataset + '/rgbLiDAR/' + label)
        edge_mask_path = os.path.normpath('../dataset/MapAI/preprocessed_' + dataset + '/edge_mask/' + label)
        original_mask_path = os.path.normpath('../dataset/MapAI/' + dataset + '/mask')
        original_image_path = os.path.normpath('../dataset/MapAI/' + dataset + '/image')
        original_lidar_path = os.path.normpath('../dataset/MapAI/' + dataset + '/lidar')

        if not os.path.exists(mask_path):
            os.makedirs(mask_path)
        if not os.path.exists(image_path):
            os.makedirs(image_path)
        if not os.path.exists(rgblidar_path):
            os.makedirs(rgblidar_path)
        if not os.path.exists(edge_mask_path):
            os.makedirs(edge_mask_path)

        # Upscale images to 512x512 and save them in new folder
        with os.scandir(original_mask_path) as entries:
            for entry in entries:
                filename = entry.name.split(".")[0]

                mask_img = cv.imread(os.path.normpath(original_mask_path + '/' + entry.name), cv.IMREAD_GRAYSCALE)
                mask_img[mask_img == 255] = 1

                # Resize images
                # Mask
                resize_mask_img = cv.resize(mask_img, (new_size, new_size), interpolation=cv.INTER_AREA)
                cv.imwrite(os.path.normpath(mask_path + '/' + entry.name), resize_mask_img)

                # Edge mask
                edge_mask = _mask_to_boundary(resize_mask_img)
                edge_mask[edge_mask == 255] = 1
                cv.imwrite(os.path.normpath(edge_mask_path + '/' + entry.name), edge_mask)

                # Image
                img = cv.imread(os.path.normpath(original_image_path + '/' + entry.name), cv.IMREAD_COLOR)
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                resize_img = cv.resize(img, (new_size, new_size), interpolation=cv.INTER_AREA)
                resize_img = cv.cvtColor(resize_img, cv.COLOR_BGR2RGB)
                cv.imwrite(os.path.normpath(image_path + '/' + entry.name), resize_img)

                # LiDAR RGB -> Concat aerial image and lidar data
                if dataset != 'task1_test':
                    lidar_data = np.load(os.path.normpath(original_lidar_path + '/' + filename + '.npy'))
                    resize_lidar = cv.resize(lidar_data, (new_size, new_size), interpolation=cv.INTER_AREA)
                    resize_lidar = np.expand_dims(resize_lidar, axis=-1)
                    rgb_lidar = np.concatenate((resize_img, resize_lidar), axis=-1)
                    imwrite(rgblidar_path + '/' + filename + '.tif', rgb_lidar.astype(np.uint8))


if __name__ == "__main__":
    preprocess()
