import os
import cv2 as cv
from tqdm import tqdm


# Preprocess Dataset

# MapAI uses images of 500x500 pixels. This input wont work on a U-Net networks as when we are Max-pooling the results
# will be: 500->250->125->62->... This 62 will later then be sampled up to 124 which will be a mismatch with 125. For
# the U-Net part of the thesis we will therefore scale up the training, validation and test images to 512x512.
# Due to class imbalance we will also remove images that contains less then a threshold of class 1.


# Defining sets that has to be rescaled
datasets = ['train', 'validation', 'task1_test']

# Define image size
original_size = 500
new_size = 512

for dataset in tqdm(datasets):
    # Make directory for preprocessed dataset
    dataset_path = os.path.normpath('dataset/MapAI/512x512_' + dataset)
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    # Make directory for preprocessed subsets
    mask_path = os.path.normpath('dataset/MapAI/512x512_' + dataset + '/mask')
    image_path = os.path.normpath('dataset/MapAI/512x512_' + dataset + '/image')
    original_mask_path = os.path.normpath('dataset/MapAI/' + dataset + '/mask')
    original_image_path = os.path.normpath('dataset/MapAI/' + dataset + '/image')

    if not os.path.exists(mask_path):
        os.makedirs(mask_path)
    if not os.path.exists(image_path):
        os.makedirs(image_path)

    # Upscale images to 512x512 -> Ignore images from training set with less than a specific threshold of building pixels
    THRESHOLD = 0.05
    with os.scandir(original_mask_path) as entries:
        for entry in entries:
            mask_img = cv.imread(os.path.normpath(original_mask_path + '/' + entry.name), cv.IMREAD_GRAYSCALE)
            mask_img[mask_img == 255] = 1
            mask_pixels = np.unique(mask_img, return_counts=True)[1] # [Class 0(Background), Class 1 (Buildings)]

            # Filter out training images
            if dataset == 'train':
                if len(mask_pixels) == 1: # Mask only contains one class
                    continue
                if mask_pixels[1]/mask_pixels[0] < THRESHOLD:
                    continue

            # Resize images
            # Mask
            resize_mask_img = cv.resize(mask_img, (new_size, new_size), interpolation = cv.INTER_AREA)
            cv.imwrite(os.path.normpath(mask_path + '/' + entry.name), resize_mask_img)

            # Image
            img = cv.imread(os.path.normpath(original_image_path + '/' + entry.name), cv.IMREAD_COLOR)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            resize_img = cv.resize(img, (new_size, new_size), interpolation = cv.INTER_AREA)
            resize_img = cv.cvtColor(resize_img, cv.COLOR_BGR2RGB)
            cv.imwrite(os.path.normpath(image_path + '/' + entry.name), resize_img)