# Imports
import os

import cv2 as cv
import tensorflow as tf
from tensorflow.keras import callbacks, preprocessing
from tqdm import tqdm
import numpy as np

import UNet

# Change GPU setting
# Limit number of GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

# Limit GPU memory
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config = config)


# Compile model
model = UNet.unet()
model.summary()


# Finding the number of images in each dataset
train_path = os.path.normpath('../dataset/MapAI/512x512_train/image')
no_train_images = len([name for name in os.listdir(train_path) if os.path.isfile(os.path.join(train_path, name))])

validation_path = os.path.normpath('../dataset/MapAI/512x512_validation/image')
no_val_images = len([name for name in os.listdir(validation_path) if os.path.isfile(os.path.join(validation_path, name))])

# Defining size of images
IMG_HEIGHT = 512
IMG_WIDTH = 512

# Creating NumPy arrays for the different subsets
print("Importing data from training and validation set")
X_train = np.zeros((no_train_images, IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
Y_train = np.zeros((no_train_images, IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)

X_val = np.zeros((no_val_images, IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
Y_val = np.zeros((no_val_images, IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)

# Defining sets
datasets = ['train', 'validation']

# Adding images to NumPy arrays
for dataset in tqdm(datasets):
    img_path = os.path.normpath('../dataset/MapAI/512x512_' + dataset + '/image')
    mask_path = os.path.normpath('../dataset/MapAI/512x512_' + dataset + '/mask')
    with os.scandir(img_path) as entries:
        for n, entry in enumerate(entries):
            img = cv.imread(os.path.normpath(img_path + '/' + entry.name))
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            mask = cv.imread(os.path.normpath(mask_path + '/' + entry.name))
            mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
            if dataset == 'train':
                X_train[n] = img
                Y_train[n] = mask
            elif dataset == 'validation':
                X_val[n] = img
                Y_val[n] = mask


# Print the size of the different sets
print('X_train size: ' + str(len(X_train)))
print('Y_train size: ' + str(len(Y_train)))
print('X_validation size: ' + str(len(X_val)))
print('Y_validation size: ' + str(len(Y_val)))


# Augmenting data
# Define seed and arguments for data generator
print("Augmenting data to get a larger dataset")

seed = 420

img_data_gen_args = dict(horizontal_flip=True, vertical_flip=True)

mask_data_gen_args = dict(horizontal_flip=True, vertical_flip=True, preprocessing_function = lambda x: np.where(x>0, 1, 0).astype(x.dtype))

# Create data generator
image_data_generator = preprocessing.image.ImageDataGenerator(**img_data_gen_args)
image_data_generator.fit(X_train, augment=True, seed=seed)
image_generator = image_data_generator.flow(X_train, seed=seed, batch_size=1, shuffle=False)

mask_data_generator = preprocessing.image.ImageDataGenerator(**mask_data_gen_args)
mask_data_generator.fit(np.expand_dims(Y_train, axis = -1), augment=True, seed=seed)
mask_generator = mask_data_generator.flow(np.expand_dims(Y_train, axis = -1), seed=seed, batch_size=1, shuffle=False)

# From image generator to numpy array
X_train_augmented = np.concatenate([image_generator.next().astype(np.uint8) for i in range(image_generator.__len__())])
Y_train_augmented = np.concatenate([mask_generator.next() for i in range(mask_generator.__len__())])

# Add augmented images to training set
X_train = np.concatenate((X_train, X_train_augmented))
Y_train = np.concatenate((Y_train, np.squeeze(Y_train_augmented)))

print('X_train size after data augmentation: ' + str(len(X_train)))
print('Y_train size after data augmentation: ' + str(len(Y_train)))


# Train Model
# Create models directory if it doesnt exist
print("Training model")
dataset_path = os.path.normpath("../models")
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

# Create callback for model.
# ModelCheckpoint -> Creates checkpoints after each epoch
# EarlyStopping -> Stops the training of the model if it doesnt improve after some epochs
callback_list = [
    callbacks.ModelCheckpoint(os.path.normpath('../models/MapAI_UNet_Task1_Checkpoint.h5'), verbose = 1, save_best_only=True),
    callbacks.EarlyStopping(monitor = 'val_loss', patience = 3)
]

# Train the model
results = model.fit(X_train, Y_train, batch_size = 8, epochs = 100, callbacks = callback_list, validation_data = (X_val, Y_val))

# Save model
print("Saving model")
model_name = input("Save model as: ")
model.save(os.path.normpath('../models/' + model_name))

