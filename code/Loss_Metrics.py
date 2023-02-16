# Imports
import tensorflow as tf
from tensorflow.keras import backend, losses
import numpy as np


# Defining metric functions
def jaccard_coef(y_true, y_pred):
    y_true_f = backend.flatten(y_true)
    y_true_f = tf.cast(y_true_f, tf.float32) # Convert the true labels to float32
    y_pred_f = backend.flatten(y_pred)
    intersection = backend.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (backend.sum(y_true_f) + backend.sum(y_pred_f) - intersection + 1.0)


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = backend.flatten(y_true)
    y_true_f = tf.cast(y_true_f, tf.float32) # Convert the true labels to float32
    y_pred_f = backend.flatten(y_pred)
    intersection = backend.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (backend.sum(y_true_f) + backend.sum(y_pred_f) + smooth)
    return dice


# Defining loss functions
def jaccard_coef_loss(y_true, y_pred):
    return 1 - jaccard_coef(y_true, y_pred)


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def BinaryCrossEntropy(y_true, y_pred):
    y_pred = backend.clip(y_pred, backend.epsilon(), 1 - backend.epsilon())
    term_0 = (1 - y_true) * backend.log(1 - y_pred + backend.epsilon())
    term_1 = y_true * backend.log(y_pred + backend.epsilon())
    return -backend.mean(term_0 + term_1, axis=0)

def binary_cross_iou(y_true, y_pred):
    weight = 0.3
    bce = losses.BinaryCrossentropy()
    return ((1 - weight) * bce(y_true, y_pred).eval()) - (weight * backend.log(jaccard_coef(y_true, y_pred)))
