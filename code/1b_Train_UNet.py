# Imports
import os
import tensorflow as tf
from tensorflow.keras import callbacks, preprocessing, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

import UNet
import CTUNet
from Loss_Metrics import jaccard_coef, jaccard_coef_loss, dice_coef_loss


# Change GPU setting
# Limit number of GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = '7'


# Limit GPU memory
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config = config)


# Compile model
print('Pick type of model to train')
models = ['U-Net', 'EfficientNetB4 U-Net', 'EfficientNetB4 CT-UNet']
for i in range(len(models)):
    print(str(i + 1) + ': ' + models[i])
model_selector = input('Which model do you want to train?: ')

model = None
if model_selector == '1':
    model = UNet.unet()
elif model_selector == '2':
    model = UNet.EfficientNetB4_unet()
elif model_selector == '3':
    model = CTUNet.EfficientNetB4_CTUnet()

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
image_generator = image_data_generator.flow_from_directory(os.path.normpath('../dataset/MapAI/512x512_train/image'),
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           seed=seed,
                                                           batch_size=batch_size,
                                                           class_mode=None)  #Very important to set this otherwise it returns multiple numpy arrays
                                                                            #thinking class mode is binary.

mask_data_generator = ImageDataGenerator(**mask_data_gen_args)
mask_generator = mask_data_generator.flow_from_directory(os.path.normpath('../dataset/MapAI/512x512_train/' + mask),
                                                         target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                         seed=seed,
                                                         batch_size=batch_size,
                                                         color_mode = 'grayscale',   #Read masks in grayscale
                                                         class_mode=None)

val_data_generator = ImageDataGenerator()
valid_img_generator = val_data_generator.flow_from_directory(os.path.normpath('../dataset/MapAI/512x512_validation/image'),
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

num_train_imgs = len(os.listdir(os.path.normpath('../dataset/MapAI/512x512_train/image/train')))
num_val_imgs = len(os.listdir(os.path.normpath('../dataset/MapAI/512x512_validation/image/val')))
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
    callbacks.EarlyStopping(monitor='val_loss', patience=5),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=3, verbose=1)
]

# Train the model
results = model.fit(train_generator, steps_per_epoch=train_steps_per_epoch, epochs=100, callbacks=callback_list, validation_data=val_generator, validation_steps=val_steps_per_epoch)

# Save model
print("Saving model")
model_name = input("Save model as: ")
model.save(os.path.normpath('../models/' + model_name))
