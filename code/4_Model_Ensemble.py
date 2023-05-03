import os
import tensorflow as tf
from tensorflow.keras import models
from tqdm import tqdm
import numpy as np
from itertools import combinations

from eval_functions import calculate_score
from Loss_Metrics import jaccard_coef, jaccard_coef_loss, dice_coef_loss
from utils import prepare_test_dataset, tta


def test_models(model_names, X_test, tta_input, task_input):
    """
    Predicts on a set of models.
    Args:
        model_names: List of names of the models that should be tested.
        X_test: Images to predict.
        tta_input: 1 if test time augmentation should be performed. Else the images are predicted without tta.
        task_input: Either 1 or 2. 1: Task 1, 2: Task 2.
    Returns:
        A dictionary of predictions from the models.
    """
    # Pick subfolder
    if task_input == '1':
        subfolder = 'task1'
    elif task_input == '2':
        subfolder = 'task2'
    else:
        raise Exception("Pick valid task")

    preds = {}
    for model_name in model_names:
        model = models.load_model(os.path.normpath('../models/' + subfolder + '/' + model_name), custom_objects={'dice_coef_loss': dice_coef_loss, 'jaccard_coef': jaccard_coef})
        # Predicting model
        if tta_input == '1':
            Y_pred = []
            for image in tqdm(X_test):
                predicition = tta(model, image)
                Y_pred.append(predicition)
        else:
            Y_pred = model.predict(X_test)

        preds[model_name] = np.array(Y_pred)

    return preds


def ensemble_models(preds, Y_test, threshold):
    """
    Tries every possible combination of weights for a set of 3 predictions. Prints the best weight combination, IoU,
    BIoU and score.
    Args:
        preds: List of predictions from a set of 3 models.
        Y_test: Ground truth.
        threshold: Pixel value threshold that should be used when determining if a pixel is a building or background.
    Returns:
        Best weight combination and resulting metrics.
    """
    iter_range = list(np.linspace(0, 1, 11))
    max_score = {'score': 0, 'iou': 0, 'biou': 0}
    best_w = []

    for w1 in tqdm(iter_range):
        for w2 in iter_range:
            for w3 in iter_range:
                if w1 + w2 + w3 != 1:
                    continue
                weights = [w1, w2, w3]
                weighted_preds = np.tensordot(preds, weights, axes=(0, 0))
                score = calculate_score(np.squeeze((weighted_preds > threshold), -1).astype(np.uint8), Y_test)
                if score['score'] > max_score['score']:
                    max_score = score
                    best_w = weights
                break

    print('Best score achieved with weights: ', best_w, ' Score: ', max_score)
    return max_score, best_w


def mass_ensemble(model_names, preds, Y_test, threshold):
    """
    Tries every possible 3 model combination of multiple models. Print the top 10 ensemble and their weights based on
    the score metric.
    Args:
        model_names: List of model names
        preds: Dictionary of predictions.
        Y_test: Ground truth.
        threshold: Pixel value threshold that should be used when determining if a pixel is a building or background.
    """
    results = {}
    for ensemble in set(combinations(model_names, 3)):
        print('Now testing ensemble for models: ', ensemble)
        current_pred = np.array([preds[ensemble[0]], preds[ensemble[1]], preds[ensemble[2]]])
        max_score, best_w = ensemble_models(current_pred, Y_test, threshold)
        results[max_score['score']] = {'models': ensemble, 'weights': best_w, 'score': max_score}

    print('Finished mass ensemble. Printing out top 10 ensembles')
    des_keys = sorted(results.keys(),  reverse=True)
    for i in range(len(des_keys)):
        res = results[des_keys[i]]
        print(i+1, ' - Models: ', res['models'], ' Weights:', res['models'], ' Score: ', res['score'])
        if i == 9:
            break


if __name__ == "__main__":
    # Select GPU
    gpu_selector = input('Which GPU do you want to use?: ')

    # Limit number of GPUs
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_selector

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

    X_test, Y_test = prepare_test_dataset(task_selector, mask_selector)

    # Select if test time augmentation should be used
    print('Enable TTA? ')
    print('1: Yes ')
    print('Otherwise: No ')
    tta_selector = input('TTA: ')

    if tta_selector == '1':
        thresh = 0.3
    else:
        thresh = 0.5

    # Select model that should be tested
    print('What type of ensemble do you want to do?')
    print('1: Ensemble with set weights')
    print('2: Find best weights from a set of 3 models')
    print('3: Find best 3 model ensemble using multiple model')
    ensemble_selector = input('Ensemble type: ')

    if ensemble_selector == '1' or ensemble_selector == '3':
        print('Enter the name of the model you want to test')
        model1_name = input("Name of model 1: ")
        model2_name = input("Name of model 2: ")
        model3_name = input("Name of model 3: ")
        model_names = [model1_name, model2_name, model3_name]
    elif ensemble_selector == '2':
        print('Enter the name of the model you want to test seperated with a comma(,)')
        model_names = input("Name of models: ").split(',')
    else:
        raise Exception("Pick valid ensemble type")

    # Predict all models
    preds = test_models(model_names, X_test, tta_selector, task_selector)

    # Ensemble
    if ensemble_selector == '1':
        pass
    elif ensemble_selector == '2':
        pass
    elif ensemble_selector == '3':
        mass_ensemble(model_names, preds, Y_test, thresh)

    print('Ensemble finished')
