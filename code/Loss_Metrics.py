# Imports
import tensorflow as tf
from tensorflow.keras import backend


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
