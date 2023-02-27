# Imports
import os
import cv2 as cv
from tqdm import tqdm
from eval_functions import _mask_to_boundary
import numpy as np


# Preprocess Dataset

# MapAI uses images of 500x500 pixels. This input wont work on a U-Net networks as when we are Max-pooling the results
# will be: 500->250->125->62->... This 62 will later then be sampled up to 124 which will be a mismatch with 125. For
# the U-Net part of the thesis we will therefore scale up the training, validation and test images to 512x512.
# Due to class imbalance we will also remove images that contains less then a threshold of class 1.


# Defining sets that has to be rescaled
datasets = ['train', 'validation', 'task1_test', 'task2_test']

# Define image size
original_size = 500
new_size = 512

for dataset in tqdm(datasets):
    # Make directory for preprocessed dataset
    dataset_path = os.path.normpath('dataset/MapAI/512x512_' + dataset)
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    # Make directory for preprocessed subsets
    mask_path = os.path.normpath('../dataset/MapAI/512x512_' + dataset + '/mask')
    image_path = os.path.normpath('../dataset/MapAI/512x512_' + dataset + '/image')
    rgblidar_path = os.path.normpath('../dataset/MapAI/512x512_' + dataset + '/rgbLiDAR')
    edge_mask_path = os.path.normpath('../dataset/MapAI/512x512_' + dataset + '/edge_mask')
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

    # Upscale images to 512x512
    with os.scandir(original_mask_path) as entries:
        for entry in entries:
            filename = entry.name.split(".")[0]

            # Change values of mask pixels
            mask_img = cv.imread(os.path.normpath(original_mask_path + '/' + entry.name), cv.IMREAD_GRAYSCALE)
            mask_img[mask_img == 255] = 1

            # Resize images
            # Mask
            resize_mask_img = cv.resize(mask_img, (new_size, new_size), interpolation = cv.INTER_AREA)
            cv.imwrite(os.path.normpath(mask_path + '/' + entry.name), resize_mask_img)

            # Edge mask
            edge_mask = _mask_to_boundary(resize_mask_img)
            edge_mask[edge_mask == 255] = 1
            cv.imwrite(os.path.normpath(edge_mask_path + '/' + entry.name), edge_mask)

            # Image
            img = cv.imread(os.path.normpath(original_image_path + '/' + entry.name), cv.IMREAD_COLOR)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            resize_img = cv.resize(img, (new_size, new_size), interpolation = cv.INTER_AREA)
            resize_img = cv.cvtColor(resize_img, cv.COLOR_BGR2RGB)
            cv.imwrite(os.path.normpath(image_path + '/' + entry.name), resize_img)

            # LiDAR RGB -> Concat aerial image and lidar data
            lidar_data = np.load(os.path.normpath(original_lidar_path + '/' + filename + '.npy'))
            resize_lidar = cv.resize(lidar_data, (new_size, new_size), interpolation = cv.INTER_AREA)
            resize_lidar = np.expand_dims(resize_lidar, axis=-1)
            rgb_lidar = np.concatenate((resize_img, resize_lidar), axis=-1)
            np.save(os.path.normpath(rgblidar_path + '/' + filename), rgb_lidar)
