# Imports
import os

import cv2 as cv
import tensorflow as tf
from tensorflow.keras import models
from tqdm import tqdm
import numpy as np

from eval_functions import calculate_score
from Loss_Metrics import jaccard_coef, jaccard_coef_loss, dice_coef_loss, binary_cross_iou

from tifffile import imwrite, imread


# Change GPU setting
# Limit number of GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

# Limit GPU memory
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config = config)

# Preparing test data
print('Select test set')
print('1: RGB')
print('2: RGBLiDAR')
train_selector = input('Which set do you want to use?: ')

train_set = None
input_shape = None
folder_name = None
NUM_CHAN = None

if train_selector == '1':
    folder_name = '512x512_task1_test'
    train_set = 'image'
    input_shape = (512, 512, 3)
    NUM_CHAN = 3
elif train_selector == '2':
    folder_name = '512x512_task2_test'
    train_set = 'rgbLiDAR'
    input_shape = (512, 512, 4)
    NUM_CHAN = 4

# Finding the number of images in each dataset
img_path = os.path.normpath('../dataset/MapAI/' + folder_name + '/' + train_set)
no_test_images = len([name for name in os.listdir(img_path) if os.path.isfile(os.path.join(img_path, name))])

# Defining size of images
IMG_HEIGHT = 512
IMG_WIDTH = 512

# Creating NumPy arrays for the different subsets
X_test = np.zeros((no_test_images, IMG_HEIGHT, IMG_WIDTH, NUM_CHAN), dtype=np.uint8)
Y_test = np.zeros((no_test_images, IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)

# Let user choose to test edge mask or building mask
print('Select mask set')
print('1: Building Masks')
print('2: Edge Masks')
mask_selector = input('Which mask set do you want to use?: ')
mask = None
if mask_selector == '1':
    mask = 'mask'
elif mask_selector == '2':
    mask = 'edge_mask'


# Adding images to NumPy arrays
mask_path = os.path.normpath('../dataset/MapAI/' + folder_name + '/' + mask)
with os.scandir(img_path) as entries:
    for n, entry in enumerate(entries):
        filename = entry.name.split(".")[0]
        img = imread(os.path.normpath(img_path + '/' + entry.name))
        #img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        X_test[n] = img
        mask = cv.imread(os.path.normpath(mask_path + '/' + filename + '.png'))
        mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
        Y_test[n] = mask

# Print the size of the different sets
print('X_train size: ' + str(len(X_test)))
print('Y_train size: ' + str(len(Y_test)))



# Testing model
# Load model
print('Test model')
model_name = input("Name of model: ")
model = models.load_model(os.path.normpath('../models/' + model_name), custom_objects={'dice_coef_loss': dice_coef_loss, 'jaccard_coef': jaccard_coef})

print("Enable TTA? ")
print("1: Yes ")
print("Otherwise: No ")
tta_input = input("TTA: ")

if tta_input == '1':     # Test time augmentation
    threshold = 0.3
    Y_pred = []
    for image in tqdm(X_test):
        prediction_original = model.predict(np.expand_dims(image, axis=0), verbose=0)[0]

        prediction_lr = model.predict(np.expand_dims(np.fliplr(image), axis=0), verbose=0)[0]
        prediction_lr = np.fliplr(prediction_lr)

        prediction_ud = model.predict(np.expand_dims(np.flipud(image), axis=0), verbose=0)[0]
        prediction_ud = np.flipud(prediction_ud)

        prediction_lr_ud = model.predict(np.expand_dims(np.fliplr(np.flipud(image)), axis=0), verbose=0)[0]
        prediction_lr_ud = np.fliplr(np.flipud(prediction_lr_ud))

        predicition = (prediction_original + prediction_lr + prediction_ud + prediction_lr_ud) / 4
        Y_pred.append(predicition)
else:
    threshold = 0.5
    Y_pred = model.predict(X_test)

# Evaluating model
score = calculate_score(np.squeeze((Y_pred > threshold), -1).astype(np.uint8), Y_test)
print(score)
