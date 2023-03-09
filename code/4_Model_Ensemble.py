import os
import tensorflow as tf
from tensorflow.keras import models
from tqdm import tqdm
import numpy as np

from eval_functions import calculate_score
from Loss_Metrics import jaccard_coef, jaccard_coef_loss, dice_coef_loss
from utils import prepare_test_dataset, tta


def test_models(model_names, X_test, tta_input):
    """
    Predicts on a set of 3 models.
    Args:
        model_names: List of names of the 3 models that should be tested.
        X_test: Images to predict.
        tta_input: 1 if test time augmentation should be performed. Else the images are predicted without tta.
    Returns:
        The prediction of the models and the threshold that should be used for evaluating.
    """
    # Loading models
    model1 = models.load_model(os.path.normpath('../models/' + model_names[0]), custom_objects={'dice_coef_loss': dice_coef_loss, 'jaccard_coef': jaccard_coef})
    model2 = models.load_model(os.path.normpath('../models/' + model_names[1]), custom_objects={'dice_coef_loss': dice_coef_loss, 'jaccard_coef': jaccard_coef})
    model3 = models.load_model(os.path.normpath('../models/' + model_names[2]), custom_objects={'dice_coef_loss': dice_coef_loss, 'jaccard_coef': jaccard_coef})
    model = [model1, model2, model3]

    # Predicting models
    if tta_input == '1':
        threshold = 0.3
        preds = []
        for m in model:
            Y_pred = []
            for image in tqdm(X_test):
                predicition = tta(m, image)
                Y_pred.append(predicition)
            preds.append(Y_pred)
        preds = np.array(preds)
    else:
        threshold = 0.5
        pred1 = model1.predict(X_test)
        pred2 = model2.predict(X_test)
        pred3 = model3.predict(X_test)
        preds = np.array([pred1, pred2, pred3])
    return preds, threshold


def ensemble_models(preds, Y_test, threshold):
    """
    Tries every possible combination of weights for a set of 3 predictions. Prints the best weight combination, IoU,
    BIoU and score.
    Args:
        preds: List of predictions from a set of 3 models.
        Y_test: Ground truth.
        threshold: Pixel value threshold that should be used when determining if a pixel is a building or background.
    """
    iter_range = list(np.linspace(0, 1, 11))
    max_score = {'score': 0, 'iou': 0, 'biou': 0}
    best_w = []

    for w1 in iter_range:
        for w2 in iter_range:
            for w3 in iter_range:
                if w1 + w2 + w3 != 1:
                    continue
                weights = [w1, w2, w3]
                weighted_preds = np.tensordot(preds, weights, axes=(0, 0))
                score = calculate_score(np.squeeze((weighted_preds > threshold), -1).astype(np.uint8), Y_test)
                print("Now predciting for weights :", w1, w2, w3, " : Score = ", score)
                if score['score'] > max_score['score']:
                    max_score = score
                    best_w = weights
                break

    print('Best score achieved with weights: ', best_w, ' Score: ', max_score)


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
    model1_name = input("Name of model 1: ")
    model2_name = input("Name of model 2: ")
    model3_name = input("Name of model 3: ")
    model_names = [model1_name, model2_name, model3_name]

    # Model ensemble
    X_test, Y_test = prepare_test_dataset(task_selector, mask_selector)
    preds, thresh = test_models(model_names, X_test, tta_selector)
    ensemble_models(preds, Y_test, thresh)
    print('Ensemble finished')
