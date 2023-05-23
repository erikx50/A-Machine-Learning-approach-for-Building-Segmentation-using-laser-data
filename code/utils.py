import os
import cv2 as cv
import numpy as np
from tifffile import imread


def prepare_test_dataset(task_input, mask_input):
    """
    Loads the test dataset into NumPY arrays.
    Args:
        task_input: Either 1 or 2. 1: Task 1, 2: Task 2.
        mask_input: Either 1 or 2. 1: Building masks, 2: Edge masks.
    Returns:
        Test images and their corresponding masks.
    """
    # Picks the dataset that should be used for testing
    if task_input == '1':
        folder_name = 'preprocessed_task1_test'
        test_set = 'image'
        NUM_CHAN = 3
    elif task_input == '2':
        folder_name = 'preprocessed_task2_test'
        test_set = 'rgbLiDAR'
        NUM_CHAN = 4
    else:
        raise Exception('Pick either RGB or RGBLiDAR')

    # Finding the number of images in each dataset
    img_path = os.path.normpath('../dataset/MapAI/' + folder_name + '/' + test_set)
    #img_path = os.path.normpath('dataset/MapAI/' + folder_name + '/' + test_set)
    no_test_images = len([name for name in os.listdir(img_path) if os.path.isfile(os.path.join(img_path, name))])

    # Creating NumPy arrays for the different subsets
    X_test = np.zeros((no_test_images, 512, 512, NUM_CHAN), dtype=np.uint8)
    Y_test = np.zeros((no_test_images, 512, 512), dtype=np.uint8)

    # Select mask set
    if mask_input == '1':
        mask = 'mask'
    elif mask_input == '2':
        mask = 'edge_mask'
    else:
        raise Exception('Pick either Building or Edge mask')

    # Adding images to NumPy arrays
    #mask_path = os.path.normpath('dataset/MapAI/' + folder_name + '/' + mask)
    mask_path = os.path.normpath('../dataset/MapAI/' + folder_name + '/' + mask)
    with os.scandir(img_path) as entries:
        for n, entry in enumerate(entries):
            filename = entry.name.split(".")[0]
            if test_set == 'image':
                img = cv.imread(os.path.normpath(img_path + '/' + entry.name))
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            else:
                img = imread(os.path.normpath(img_path + '/' + entry.name))
            X_test[n] = img
            mask = cv.imread(os.path.normpath(mask_path + '/' + filename + '.PNG'))
            mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
            Y_test[n] = mask

    # Print the size of the different sets
    print('X_train size: ' + str(len(X_test)))
    print('Y_train size: ' + str(len(Y_test)))
    return X_test, Y_test


def tta(model, image):
    """
    Performs test time augmentation on an image.
    Args:
        model: The model that should predict the image.
        image: Image to predict.
    Returns:
        The average prediction of the test time augmentation.
    """
    prediction_original = model.predict(np.expand_dims(image, axis=0), verbose=0)[0]

    prediction_lr = model.predict(np.expand_dims(np.fliplr(image), axis=0), verbose=0)[0]
    prediction_lr = np.fliplr(prediction_lr)

    prediction_ud = model.predict(np.expand_dims(np.flipud(image), axis=0), verbose=0)[0]
    prediction_ud = np.flipud(prediction_ud)

    prediction_lr_ud = model.predict(np.expand_dims(np.fliplr(np.flipud(image)), axis=0), verbose=0)[0]
    prediction_lr_ud = np.fliplr(np.flipud(prediction_lr_ud))

    predicition = (prediction_original + prediction_lr + prediction_ud + prediction_lr_ud) / 4
    return predicition
