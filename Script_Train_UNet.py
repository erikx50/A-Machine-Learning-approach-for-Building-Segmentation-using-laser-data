# Imports
import os

import cv2 as cv
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers, callbacks, losses
from tqdm import tqdm
import numpy as np


# Change GPU setting
# Limit number of GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = '4'

# Limit GPU memory
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config = config)


# Create and Compile U-Net
# Creating the model
def unet(input_size=(512, 512, 3)):
    # Encoder Part
    # Layer 1
    inputs = layers.Input(input_size)
    inputs_rescaled = layers.Lambda(lambda x: x / 255)(inputs) # Rescale input pixel values to floating point values
    c1 = layers.Conv2D(64, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(inputs_rescaled)
    #c1 = layers.Dropout(0.1)(c1)
    c1 = layers.Conv2D(64, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(c1)
    p1 = layers.MaxPooling2D((2,2))(c1)
    
    # Layer 2
    c2 = layers.Conv2D(128, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(p1)
    #c2 = layers.Dropout(0.1)(c2)
    c2 = layers.Conv2D(128, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(c2)
    p2 = layers.MaxPooling2D((2,2))(c2)
    
    # Layer 3
    c3 = layers.Conv2D(256, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(p2)
    #c3 = layers.Dropout(0.1)(c3)
    c3 = layers.Conv2D(256, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(c3)
    p3 = layers.MaxPooling2D((2,2))(c3)
    
    # Layer 4
    c4 = layers.Conv2D(512, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(p3)
    #c4 = layers.Dropout(0.1)(c4)
    c4 = layers.Conv2D(512, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(c4)
    p4 = layers.MaxPooling2D((2,2))(c4)
    
    # Layer 5
    c5 = layers.Conv2D(1024, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(p4)
    #c5 = layers.Dropout(0.1)(c5)
    c5 = layers.Conv2D(1024, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(c5)
    
    
    # Decoder Part
    # Layer 6
    u6 = layers.Conv2DTranspose(512, (2,2), strides=(2,2), padding = 'same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(512, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(u6)
    #c6 = layers.Dropout(0.1)(c6)
    c6 = layers.Conv2D(512, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(c6)
    
    # Layer 7
    u7 = layers.Conv2DTranspose(256, (2,2), strides=(2,2), padding = 'same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(256, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(u7)
    #c7 = layers.Dropout(0.1)(c7)
    c7 = layers.Conv2D(256, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(c7)
    
    # Layer 8
    u8 = layers.Conv2DTranspose(128, (2,2), strides=(2,2), padding = 'same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(128, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(u8)
    #c8 = layers.Dropout(0.1)(c8)
    c8 = layers.Conv2D(128, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(c8)
    
    # Layer 9
    u9 = layers.Conv2DTranspose(64, (2,2), strides=(2,2), padding = 'same')(c8)
    u9 = layers.concatenate([u9, c1], axis=3)
    c9 = layers.Conv2D(64, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(u9)
    #c9 = layers.Dropout(0.1)(c9)
    c9 = layers.Conv2D(64, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(c9)
    
    outputs = layers.Conv2D(1, (1,1), activation = 'sigmoid')(c9)
    
    # Compiling model
    model = models.Model([inputs], [outputs])
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    return model


model = unet()
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


# Train the model - validation_data = (X_val, Y_val)
results = model.fit(X_train, Y_train, batch_size = 4, epochs = 25, callbacks = callback_list, validation_split = 0.2)

# Save model
model.save(os.path.normpath('models/unet1'))

