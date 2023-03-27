import os
import tensorflow as tf
from tensorflow.keras import models
from tqdm import tqdm
import numpy as np

from eval_functions import calculate_score
from Loss_Metrics import jaccard_coef, jaccard_coef_loss, dice_coef_loss
from utils import prepare_test_dataset, tta


def test_model(model_name, X_test, Y_test, tta_input):
    """
    Tests the model and prints the IoU, BIoU and Score.
    Args:
        model_name: Name of the model that should be tested. Must be a valid name in the model folder.
        X_test: Images to predict.
        Y_test: Masks corresponding to the images being predicted.
        tta_input: 1 if test time augmentation should be performed. Else the images are predicted without tta.
    """
    # Load model
    #model = models.load_model(os.path.normpath('../models/' + model_name), custom_objects={'dice_coef_loss': dice_coef_loss, 'jaccard_coef': jaccard_coef})
    model = models.load_model(os.path.normpath('../models/' + model_name), custom_objects={'jaccard_coef': jaccard_coef})

    # Predicting model
    if tta_input == '1':
        threshold = 0.3
        Y_pred = []
        for image in tqdm(X_test):
            predicition = tta(model, image)
            Y_pred.append(predicition)
        Y_pred = np.array(Y_pred)
    else:
        threshold = 0.5
        Y_pred = model.predict(X_test)

    # Evaluating model
    score = calculate_score(np.squeeze((Y_pred > threshold), -1).astype(np.uint8), Y_test)
    print(score)


if __name__ == "__main__":
    # Limit number of GPUs
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'

    # Limit GPU memory
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)

    # Select which type of set to test on. 1: RGB, 2: RGBLiDAR
    print('Select train set')
    print('1: RGB')
    print('2: RGBLiDAR')
    task_selector = input('Which set do you want to use?: ')

    # Selecting mask set
    print('Select mask set')
    print('1: Building Masks')
    print('2: Edge Masks')
    mask_selector = input('Which mask set do you want to use?: ')

    # Select if test time augmentation should be used
    print('Enable TTA? ')
    print('1: Yes ')
    print('Otherwise: No ')
    tta_selector = input('TTA: ')

    # Select model that should be tested
    print('Enter the name of the model you want to test')
    model_name = input('Name of model: ')

    # Start testing
    X_test, Y_test = prepare_test_dataset(task_selector, mask_selector)
    test_model(model_name, X_test, Y_test, tta_selector)
    print('Testing finished')
