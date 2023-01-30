import os

import cv2 as cv
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers, callbacks, losses
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

from eval_functions import calculate_score


# Change GPU setting
# Limit number of GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = '4'

# Limit GPU memory
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config = config)

# Preparing test data
# Finding the number of images in each dataset
test_path = os.path.normpath('dataset/MapAI/512x512_task1_test/image')
no_test_images = len([name for name in os.listdir(test_path) if os.path.isfile(os.path.join(test_path, name))])

# Defining size of images
IMG_HEIGHT = 512
IMG_WIDTH = 512

# Creating NumPy arrays for the different subsets
X_test = np.zeros((no_test_images, IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
Y_test = np.zeros((no_test_images, IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)

# Defining sets
subsets = ['image', 'mask']

# Adding images to NumPy arrays
for subset in tqdm(subsets):
    subset_path = os.path.normpath('dataset/MapAI/512x512_task1_test/' + subset)
    with os.scandir(subset_path) as entries:
        for n, entry in enumerate(entries):
            img = cv.imread(os.path.normpath(subset_path + '/' + entry.name))
            if subset == 'image':
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                X_test[n] = img
            if subset == 'mask':
                img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                Y_test[n] = img

# Print the size of the different sets
print('X_train size: ' + str(len(X_test)))
print('Y_train size: ' + str(len(Y_test)))



# Testing model
# Load model
model = models.load_model(os.path.normpath('models/unet1'))
Y_pred = model.predict(X_test)

# Evaluating model
score = calculate_score(np.argmax(Y_pred, -1), Y_test)
print(score)
