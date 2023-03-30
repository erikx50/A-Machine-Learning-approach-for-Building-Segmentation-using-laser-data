import os
import tensorflow as tf
from tensorflow.keras import callbacks, preprocessing, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

import UNet
import CTUNet
from Loss_Metrics import jaccard_coef, jaccard_coef_loss, dice_coef_loss


def prepare_model(train_input, model_input):
    """
    Selects the model that should be trained.
    Args:
        train_input: Either 1 or 2. 1: Task 1, 2: Task 2.
        model_input: The model that should be trained.
    Returns:
        A U-Net model
    """
    # Prepare model to train on RGB or RGBLiDAR images
    if train_input == '1':
        input_shape = (512, 512, 3)
        weight = 'imagenet'
    elif train_input == '2':
        input_shape = (512, 512, 4)
        weight = None
    else:
        raise Exception('Pick either RGB or RGBLiDAR')

    # Select model to train
    if model_input == '1':
        model = UNet.unet(input_shape)
    elif model_input == '2':
        model = UNet.EfficientNetB4_unet(input_shape, weight)
    elif model_input == '3':
        model = UNet.EfficientNetV2S_unet(input_shape, weight)
    elif model_input == '4':
        model = UNet.ResNet50V2_unet(input_shape, weight)
    elif model_input == '5':
        model = UNet.DenseNet201_unet(input_shape, weight)
    elif model_input == '6':
        model = CTUNet.EfficientNetB4_CTUnet(input_shape, weight)
    elif model_input == '7':
        model = CTUNet.EfficientNetV2S_CTUnet(input_shape, weight)
    elif model_input == '8':
        model = CTUNet.ResNet50V2_CTUnet(input_shape, weight)
    elif model_input == '9':
        model = CTUNet.DenseNet201_CTUnet(input_shape, weight)
    else:
        raise Exception('No model matching the input')

    model.summary()
    return model


def prepare_dataset_generator(train_input, mask_input, target_size=(512, 512), seed=24, batch_size=6):
    """
    Creates data generators for training and validation images and masks.
    Args:
        train_input: Either 1 or 2. 1: Task 1, 2: Task 2.
        mask_input: Either 1 or 2. 1: Building masks, 2: Edge masks.
        target_size: The pixel size of the images used to train the model.
        seed: The seed for the data generators.
        batch_size: The batch size for the generators.
    Returns:
        Training image generator, validation image generator, training image steps per epoch, validation image steps
        per epoch.
    """
    # Getting settings for data generator based on the image type used for testing
    if train_input == '1':
        train_set = 'image'
        color_mode = 'rgb'
    elif train_input == '2':
        train_set = 'rgbLiDAR'
        color_mode = 'rgba'
    else:
        raise Exception('Pick either RGB or RGBLiDAR')

    # Prepare model to use building or edge masks
    if mask_input == '1':
        mask = 'mask'
    elif mask_input == '2':
        mask = 'edge_mask'
    else:
        raise Exception('Pick either Building or Edge mask')

    # Setting augmentation args
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
                              preprocessing_function=lambda x: np.where(x > 0, 1, 0).astype(x.dtype)) # Binarize the output.

    # Creating data generator for train and validation images and masks
    image_data_generator = ImageDataGenerator(**img_data_gen_args)
    image_generator = image_data_generator.flow_from_directory(os.path.normpath('../dataset/MapAI/preprocessed_train/' + train_set),
                                                               target_size=target_size,
                                                               seed=seed,
                                                               batch_size=batch_size,
                                                               color_mode=color_mode,
                                                               class_mode=None)

    mask_data_generator = ImageDataGenerator(**mask_data_gen_args)
    mask_generator = mask_data_generator.flow_from_directory(os.path.normpath('../dataset/MapAI/preprocessed_train/' + mask),
                                                             target_size=target_size,
                                                             seed=seed,
                                                             batch_size=batch_size,
                                                             color_mode='grayscale',
                                                             class_mode=None)

    val_data_generator = ImageDataGenerator()
    valid_img_generator = val_data_generator.flow_from_directory(os.path.normpath('../dataset/MapAI/preprocessed_validation/' + train_set),
                                                                 target_size=target_size,
                                                                 seed=seed,
                                                                 batch_size=batch_size,
                                                                 color_mode=color_mode,
                                                                 class_mode=None)

    valid_mask_generator = val_data_generator.flow_from_directory(os.path.normpath('../dataset/MapAI/preprocessed_validation/' + mask),
                                                                  target_size=target_size,
                                                                  seed=seed,
                                                                  batch_size=batch_size,
                                                                  color_mode='grayscale',
                                                                  class_mode=None)

    train_generator = zip(image_generator, mask_generator)
    val_generator = zip(valid_img_generator, valid_mask_generator)
    num_train_imgs = len(os.listdir(os.path.normpath('../dataset/MapAI/preprocessed_train/' + train_set + '/train')))
    num_val_imgs = len(os.listdir(os.path.normpath('../dataset/MapAI/preprocessed_validation/' + train_set + '/val')))
    train_steps_per_epoch = num_train_imgs // batch_size
    val_steps_per_epoch = num_val_imgs // batch_size

    print('Number of train images: ' + str(num_train_imgs))
    print('Number of validation images: ' + str(num_val_imgs))
    return train_generator, val_generator, train_steps_per_epoch, val_steps_per_epoch


def train_model(model, model_name, train_input, train_generator, val_generator, train_steps_per_epoch, val_steps_per_epoch):
    """
    Trains a model, saves the model in the model folder.
    Args:
        model: The model that should be trained.
        model_name: The name the model should be saved as.
        train_input: Either 1 or 2. 1: Task 1, 2: Task 2.
        train_generator: Image generator containing training images.
        val_generator: Image generator containing validation images.
        train_steps_per_epoch: Number of training steps per epoch.
        val_steps_per_epoch: Number of validation steps per epoch.
    """
    # Create models directory if it doesnt exist
    print('Training model')
    models_path = os.path.normpath('../models')
    if not os.path.exists(models_path):
        os.makedirs(models_path)

    # Create logs directory if it doesnt exist
    logs_path = os.path.normpath('../logs')
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)

    # Setting patience for callbacks depending on the images used for training the model and creating sub-folder for models
    # depending on the task.
    if train_input == '1':
        patience = 5
        task_path = os.path.normpath('../models/task1')
        if not os.path.exists(task_path):
            os.makedirs(task_path)
    elif train_input == '2':
        patience = 5
        task_path = os.path.normpath('../models/task2')
        if not os.path.exists(task_path):
            os.makedirs(task_path)
    else:
        raise Exception('Pick either RGB or RGBLiDAR')

    # Create callback for model.
    # ModelCheckpoint -> Creates checkpoints after each epoch
    # EarlyStopping -> Stops the training of the model if it doesnt improve after some epochs
    # ReduceLROnPlateau -> Reduces learning rate after not improving val loss for some time
    # CSVLogger -> Logs the training in a CSV file
    callback_list = [
        callbacks.ModelCheckpoint(os.path.normpath(task_path + '/' + model_name + '_Checkpoint.h5'), verbose=1, save_best_only=True),
        callbacks.EarlyStopping(monitor='val_loss', patience=patience*2),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=patience, verbose=1),
        callbacks.CSVLogger(os.path.normpath(logs_path + '/' + model_name + '_log'), separator=',', append=False)
    ]

    # Train the model
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss=[dice_coef_loss], metrics=[jaccard_coef, 'accuracy'])
    model.fit(train_generator, steps_per_epoch=train_steps_per_epoch, epochs=100, callbacks=callback_list, validation_data=val_generator, validation_steps=val_steps_per_epoch, verbose=2)

    print("Saving model")
    model.save(os.path.normpath(task_path + '/' + model_name))


if __name__ == "__main__":
    # Limit number of GPUs
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'

    # Limit GPU memory
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)

    # Select which type of set to train on. 1: RGB, 2: RGBLiDAR
    print('Select train set')
    print('1: RGB')
    print('2: RGBLiDAR')
    train_selector = input('Which set do you want to use?: ')

    # Selecting mask set
    print('Select mask set')
    print('1: Building Masks')
    print('2: Edge Masks')
    mask_selector = input('Which mask set do you want to use?: ')

    # Selecting model to train
    print('Pick type of model to train')
    models = ['U-Net', 'EfficientNetB4 U-Net', 'EfficientNetV2S U-Net', 'ResNet50V2 U-Net', 'DenseNet201 U-Net',
              'EfficientNetB4 CT-UNet', 'EfficientNetV2S CT-UNet', 'ResNet50V2 CT-Unet', 'DenseNet201 CT-Unet']
    for i in range(len(models)):
        print(str(i + 1) + ': ' + models[i])
    model_selector = input('Which model do you want to train?: ')

    # Select name of model
    print('Select model name')
    name_selector = input('What is the name the model should be saved as?: ')

    # Start training
    model = prepare_model(train_selector, model_selector)
    train_generator, val_generator, train_steps_per_epoch, val_steps_per_epoch = prepare_dataset_generator(train_selector, mask_selector, batch_size=6)
    train_model(model, name_selector, train_selector, train_generator, val_generator, train_steps_per_epoch, val_steps_per_epoch)
    print('Training finished')
