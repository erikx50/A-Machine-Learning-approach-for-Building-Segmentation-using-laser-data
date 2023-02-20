# Imports
import os

import cv2 as cv
import tensorflow as tf
from tensorflow.keras import models
from tqdm import tqdm
import numpy as np

from eval_functions import calculate_score
from Loss_Metrics import jaccard_coef, jaccard_coef_loss, dice_coef_loss, binary_cross_iou

# Change GPU setting
# Limit number of GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

# Limit GPU memory
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config = config)

# Preparing test data
# Finding the number of images in each dataset
test_path = os.path.normpath('../dataset/MapAI/512x512_task1_test/image')
no_test_images = len([name for name in os.listdir(test_path) if os.path.isfile(os.path.join(test_path, name))])

# Defining size of images
IMG_HEIGHT = 512
IMG_WIDTH = 512

# Creating NumPy arrays for the different subsets
X_test = np.zeros((no_test_images, IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
Y_test = np.zeros((no_test_images, IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)


# Adding images to NumPy arrays
img_path = os.path.normpath('../dataset/MapAI/512x512_task1_test/image')
mask_path = os.path.normpath('../dataset/MapAI/512x512_task1_test/mask')
with os.scandir(img_path) as entries:
    for n, entry in enumerate(entries):
        img = cv.imread(os.path.normpath(img_path + '/' + entry.name))
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        X_test[n] = img
        mask = cv.imread(os.path.normpath(mask_path + '/' + entry.name))
        mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
        Y_test[n] = mask

# Print the size of the different sets
print('X_train size: ' + str(len(X_test)))
print('Y_train size: ' + str(len(Y_test)))


# Testing models
# Load model
print('Test model')
model1_name = input("Name of model 1: ")
model1 = models.load_model(os.path.normpath('../models/' + model1_name), custom_objects={'dice_coef_loss': dice_coef_loss, 'jaccard_coef': jaccard_coef})

model2_name = input("Name of model 2: ")
model2 = models.load_model(os.path.normpath('../models/' + model2_name), custom_objects={'dice_coef_loss': dice_coef_loss, 'jaccard_coef': jaccard_coef})

model = [model1, model2]

# Predict
pred1 = model1.predict(X_test)
pred2 = model2.predict(X_test)

preds = np.array([pred1, pred2])

iter_range = list(np.linspace(0, 1, 11))

max_score = {'score': 0, 'iou': 0, 'biou': 0}
best_w = []

for w1 in iter_range:
    for w2 in iter_range:
        if w1 + w2 != 1:
            continue
        weights = [w1, w2]
        weighted_preds = np.tensordot(preds, weights, axes=((0),(0)))
        score = calculate_score(np.squeeze((weighted_preds > 0.5), -1).astype(np.uint8), Y_test)
        print("Now predciting for weights :", w1, w2, " : Score = ", score)
        if score['score'] > max_score['score']:
            max_score = score
            best_w = weights

print('Best score achieved with weights: ', best_w, ' Score: ', max_score)
