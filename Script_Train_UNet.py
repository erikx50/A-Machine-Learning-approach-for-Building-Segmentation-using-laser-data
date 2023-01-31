# Imports
import os

import cv2 as cv
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers, callbacks, losses
from tqdm import tqdm
import numpy as np

import UNet


# Change GPU setting
# Limit number of GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# Limit GPU memory
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config = config)


# Compile model
model = UNet.unet_dropout()
model.summary()


# Finding the number of images in each dataset
train_path = os.path.normpath('dataset/MapAI/512x512_train/image')
no_train_images = len([name for name in os.listdir(train_path) if os.path.isfile(os.path.join(train_path, name))])

validation_path = os.path.normpath('dataset/MapAI/512x512_validation/image')
no_val_images = len([name for name in os.listdir(validation_path) if os.path.isfile(os.path.join(validation_path, name))])

# Defining size of images
IMG_HEIGHT = 512
IMG_WIDTH = 512

# Creating NumPy arrays for the different subsets
X_train = np.zeros((no_train_images, IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
Y_train = np.zeros((no_train_images, IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)

X_val = np.zeros((no_val_images, IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
Y_val = np.zeros((no_val_images, IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)

# Defining sets
datasets = ['train', 'validation']
subsets = ['image', 'mask']

# Adding images to NumPy arrays
for dataset in tqdm(datasets):
    dataset_path = os.path.normpath('dataset/MapAI/512x512_' + dataset)
    for subset in tqdm(subsets):
        subset_path = os.path.normpath('dataset/MapAI/512x512_' + dataset + '/' + subset)
        with os.scandir(subset_path) as entries:
            for n, entry in enumerate(entries):
                img = cv.imread(os.path.normpath(subset_path + '/' + entry.name))
                if subset == 'image':
                    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                    if dataset == 'train':
                        X_train[n] = img
                    if dataset == 'validation':
                        X_val[n] = img
                if subset == 'mask':
                    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                    if dataset == 'train':
                        Y_train[n] = img
                    if dataset == 'validation':
                        Y_val[n] = img


# Print the size of the different sets
print('X_train size: ' + str(len(X_train)))
print('Y_train size: ' + str(len(Y_train)))
print('X_validation size: ' + str(len(X_val)))
print('Y_validation size: ' + str(len(Y_val)))


# Train Model
# Create callback for model. 

# ModelCheckpoint -> Creates checkpoints after each epoch
# EarlyStopping -> Stops the training of the model if it doesnt improve after some epochs
callback_list = [
    callbacks.ModelCheckpoint(os.path.normpath('models/MapAI_UNet_Task1_Checkpoint.h5'), verbose = 1, save_best_only=True),
    callbacks.EarlyStopping(monitor = 'val_loss', patience = 3)
]

# Set class weights
class_weight = {0: 1., 1: 10.}

# Train the model
results = model.fit(X_train, Y_train, batch_size = 4, epochs = 25, callbacks = callback_list, validation_data = (X_val, Y_val), sample_weight=class_weight)

# Save model
model.save(os.path.normpath('models/recentUNet'))

