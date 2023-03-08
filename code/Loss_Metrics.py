import tensorflow as tf
from tensorflow.keras import backend, losses


# Defining metric functions
def jaccard_coef(y_true, y_pred):
    """
    Calculates the IoU from a set of true and predicted labels.
    Args:
        y_true: True labels
        y_pred: Predicted labels
    Returns:
        The IoU of the true and predicted labels.
    """
    y_true_f = backend.flatten(y_true)
    y_true_f = tf.cast(y_true_f, tf.float32)    # Convert the true labels to float32
    y_pred_f = backend.flatten(y_pred)
    intersection = backend.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (backend.sum(y_true_f) + backend.sum(y_pred_f) - intersection + 1.0)


def dice_coef(y_true, y_pred, smooth=1):
    """
    Calculates the Dice coefficient from a set of true and predicted labels.
    Args:
        y_true: True labels
        y_pred: Predicted labels
        smooth: Smoothing factor
    Returns:
        The Dice coefficient of the true and predicted labels.
    """
    y_true_f = backend.flatten(y_true)
    y_true_f = tf.cast(y_true_f, tf.float32)    # Convert the true labels to float32
    y_pred_f = backend.flatten(y_pred)
    intersection = backend.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (backend.sum(y_true_f) + backend.sum(y_pred_f) + smooth)
    return dice


# Defining loss functions
def jaccard_coef_loss(y_true, y_pred):
    """
    Calculates the IoU loss from a set of true and predicted labels.
    Args:
        y_true: True labels
        y_pred: Predicted labels
    Returns:
        The IoU loss of the true and predicted labels.
    """
    return 1 - jaccard_coef(y_true, y_pred)


def dice_coef_loss(y_true, y_pred):
    """
    Calculates the Dice coefficient loss from a set of true and predicted labels.
    Args:
        y_true: True labels
        y_pred: Predicted labels
    Returns:
        The Dice coefficient loss of the true and predicted labels.
    """
    return 1 - dice_coef(y_true, y_pred)

