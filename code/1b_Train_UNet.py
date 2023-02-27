# Imports
import os
import tensorflow as tf
from tensorflow.keras import callbacks, preprocessing, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

####
import cv2 as cv
from tqdm import tqdm
from tifffile import imwrite, imread
####

import UNet
import CTUNet
from Loss_Metrics import jaccard_coef, jaccard_coef_loss, dice_coef_loss, binary_cross_iou


# Change GPU setting
# Limit number of GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = '7'


# Limit GPU memory
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config = config)

# Select which type of set to train on. 1: RGB, 2: RGBLiDAR
print('Select train set')
print('1: RGB')
print('2: RGBLiDAR')
train_selector = input('Which set do you want to use?: ')
train_set = None
input_shape = None
weight = None
if train_selector == '1':
    train_set = 'image'
    input_shape = (512, 512, 3)
    weight = 'imagenet'
elif train_selector == '2':
    train_set = 'rgbLiDAR'
    input_shape = (512, 512, 4)

# Compile model
print('Pick type of model to train')
models = ['U-Net', 'EfficientNetB4 U-Net', 'EfficientNetB4 CT-UNet', 'EfficientNetV2S CT-UNet', 'ResNet50V2 CT-Unet', 'DenseNet201 CT-Unet']
for i in range(len(models)):
    print(str(i + 1) + ': ' + models[i])
model_selector = input('Which model do you want to train?: ')

model = None
if model_selector == '1':
    model = UNet.unet(input_shape)
elif model_selector == '2':
    model = UNet.EfficientNetB4_unet(input_shape, weight)
elif model_selector == '3':
    model = CTUNet.EfficientNetB4_CTUnet(input_shape, weight)
elif model_selector == '4':
    model = CTUNet.EfficientNetV2S_CTUnet(input_shape, weight)
elif model_selector == '5':
    model = CTUNet.ResNet50V2_CTUnet(input_shape, weight)
elif model_selector == '6':
    model = CTUNet.DenseNet201_CTUnet(input_shape, weight)

model.summary()
model.compile(optimizer=optimizers.Adam(learning_rate=0.000015), loss=[dice_coef_loss], metrics=[jaccard_coef, 'accuracy'])

# Creating data generators for training data
# Selecting mask set
print('Select mask set')
print('1: Building Masks')
print('2: Edge Masks')
mask_selector = input('Which mask set do you want to use?: ')
mask = None
if mask_selector == '1':
    mask = 'mask'
elif mask_selector == '2':
    mask = 'edge_mask'

IMG_HEIGHT = 512
IMG_WIDTH = 512

datagen = False
if datagen:
    seed = 24
    batch_size = 6

    img_data_gen_args = dict(rotation_range=90,
                          width_shift_range=0.3,
                          height_shift_range=0.3,
                          shear_range=0.5,
                          zoom_range=0.3,
                          horizontal_flip=True,
                          vertical_flip=True,
                          fill_mode='reflect')

    mask_data_gen_args = dict(rotation_range=90,
                          width_shift_range=0.3,
                          height_shift_range=0.3,
                          shear_range=0.5,
                          zoom_range=0.3,
                          horizontal_flip=True,
                          vertical_flip=True,
                          fill_mode='reflect',
                          preprocessing_function = lambda x: np.where(x>0, 1, 0).astype(x.dtype)) #Binarize the output again.


    image_data_generator = ImageDataGenerator(**img_data_gen_args)
    image_generator = image_data_generator.flow_from_directory(os.path.normpath('../dataset/MapAI/512x512_train/' + train_set),
                                                               target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                               seed=seed,
                                                               batch_size=batch_size,
                                                               class_mode=None)

    mask_data_generator = ImageDataGenerator(**mask_data_gen_args)
    mask_generator = mask_data_generator.flow_from_directory(os.path.normpath('../dataset/MapAI/512x512_train/' + mask),
                                                             target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                             seed=seed,
                                                             batch_size=batch_size,
                                                             color_mode = 'grayscale',   #Read masks in grayscale
                                                             class_mode=None)

    val_data_generator = ImageDataGenerator()
    valid_img_generator = val_data_generator.flow_from_directory(os.path.normpath('../dataset/MapAI/512x512_validation/' + train_set),
                                                                   target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                                   seed=seed,
                                                                   batch_size=batch_size,
                                                                   class_mode=None)

    valid_mask_generator = val_data_generator.flow_from_directory(os.path.normpath('../dataset/MapAI/512x512_validation/' + mask),
                                                                   target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                                   seed=seed,
                                                                   batch_size=batch_size,
                                                                   color_mode = 'grayscale',   #Read masks in grayscale
                                                                   class_mode=None)

    train_generator = zip(image_generator, mask_generator)
    val_generator = zip(valid_img_generator, valid_mask_generator)

    num_train_imgs = len(os.listdir(os.path.normpath('../dataset/MapAI/512x512_train/' + train_set + '/train')))
    num_val_imgs = len(os.listdir(os.path.normpath('../dataset/MapAI/512x512_validation/' + train_set + '/val')))
    train_steps_per_epoch = num_train_imgs // batch_size
    val_steps_per_epoch = num_val_imgs // batch_size

    print("Number of train images: " + str(num_train_imgs))
    print("Number of validation images: " + str(num_val_imgs))

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
        callbacks.ModelCheckpoint(os.path.normpath('../models/MapAI_UNet_Task1_Checkpoint.h5'), verbose=1, save_best_only=True),
        callbacks.EarlyStopping(monitor='val_loss', patience=6),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=3, verbose=1)
    ]

    # Train the model
    results = model.fit(train_generator, steps_per_epoch=train_steps_per_epoch, epochs=100, callbacks=callback_list, validation_data=val_generator, validation_steps=val_steps_per_epoch, verbose = 2)
else:
    # Finding the number of images in each dataset
    train_path = os.path.normpath('../dataset/MapAI/512x512_train/rgbLiDAR/train')
    no_train_images = len([name for name in os.listdir(train_path) if os.path.isfile(os.path.join(train_path, name))])

    validation_path = os.path.normpath('../dataset/MapAI/512x512_validation/rgbLiDAR/val')
    no_val_images = len([name for name in os.listdir(validation_path) if os.path.isfile(os.path.join(validation_path, name))])

    # Defining size of images
    IMG_HEIGHT = 512
    IMG_WIDTH = 512

    # Creating NumPy arrays for the different subsets
    X_train = np.zeros((no_train_images, IMG_HEIGHT, IMG_WIDTH, 4), dtype=np.float32)
    Y_train = np.zeros((no_train_images, IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)

    X_val = np.zeros((no_val_images, IMG_HEIGHT, IMG_WIDTH, 4), dtype=np.float32)
    Y_val = np.zeros((no_val_images, IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)

    # Defining sets
    datasets = ['train', 'validation']

    # Adding images to NumPy arrays
    for dataset in tqdm(datasets):
        label = None
        if dataset == 'train':
            label = 'train'
        elif dataset == 'validation':
            label = 'val'

        img_path = os.path.normpath('../dataset/MapAI/512x512_' + dataset + '/rgbLiDAR/' + label)
        mask_path = os.path.normpath('../dataset/MapAI/512x512_' + dataset + '/mask/' + label)
        with os.scandir(img_path) as entries:
            for n, entry in enumerate(entries):
                filename = entry.name.split(".")[0]

                img = imread(os.path.normpath(img_path + '/' + filename + '.tif'))
                mask = cv.imread(os.path.normpath(mask_path + '/' + filename + '.PNG'))
                mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
                if dataset == 'train':
                    X_train[n] = img
                    Y_train[n] = mask
                elif dataset == 'validation':
                    X_val[n] = img
                    Y_val[n] = mask
                if n == 1000:
                    break

    # Print the size of the different sets
    print('X_train size: ' + str(len(X_train)))
    print('Y_train size: ' + str(len(Y_train)))
    print('X_validation size: ' + str(len(X_val)))
    print('Y_validation size: ' + str(len(Y_val)))

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
        callbacks.ModelCheckpoint(os.path.normpath('../models/MapAI_UNet_Task1_Checkpoint.h5'), verbose=1, save_best_only=True),
        callbacks.EarlyStopping(monitor='val_loss', patience=6),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=3, verbose=1)
    ]

    # Train the model
    results = model.fit(X_train, Y_train, batch_size=1, epochs=100, callbacks=callback_list, validation_data =(X_val, Y_val))

    # Save model
print("Saving model")
model_name = input("Save model as: ")
model.save(os.path.normpath('../models/' + model_name))
