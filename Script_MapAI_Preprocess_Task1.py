import os
import cv2 as cv
from tqdm import tqdm


# Upscaling Dataset

# MapAI uses images of 500x500 pixels. This input wont work on a U-Net networks as when we are Max-pooling the results
# will be: 500->250->125->62->... This 62 will later then be sampled up to 124 which will be a mismatch with 125. For
# the U-Net part of the thesis we will therefore scale up the training, validation and test images to 512x512.
# Defining sets that has to be rescaled
datasets = ['train', 'validation', 'task1_test']
subsets = ['image', 'mask']

for dataset in tqdm(datasets):
    dataset_path = os.path.normpath('dataset/MapAI/512x512_' + dataset)
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    for subset in tqdm(subsets):
        subset_path = os.path.normpath('dataset/MapAI/512x512_' + dataset + '/' + subset)
        if not os.path.exists(subset_path):
            os.makedirs(subset_path)
        original_filepath = os.path.normpath('dataset/MapAI/' + dataset + '/' + subset)
        with os.scandir(original_filepath) as entries:
            for entry in entries:
                if subset == 'mask':
                    img = cv.imread(os.path.normpath(original_filepath + '/' + entry.name), cv.IMREAD_GRAYSCALE)
                    img[img == 255] = 1
                    resize_img = cv.resize(img, (512, 512), interpolation = cv.INTER_AREA)
                else:
                    img = cv.imread(os.path.normpath(original_filepath + '/' + entry.name), cv.IMREAD_COLOR)
                    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                    resize_img = cv.resize(img, (512, 512), interpolation = cv.INTER_AREA)
                    resize_img = cv.cvtColor(resize_img, cv.COLOR_BGR2RGB)
                cv.imwrite(os.path.normpath(subset_path + '/' + entry.name), resize_img)
