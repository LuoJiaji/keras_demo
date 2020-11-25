import numpy as np
import keras
from keras.utils import to_categorical


def smooth_labels(labels, factor=0.1):
    # smooth the labels
    labels *= (1 - factor)
    labels += (factor / labels.shape[1])
 
    # returned the smoothed labels
    return labels

labels = np.random.randint(0, 8, 20)
labels_one_hot = to_categorical(labels)
print(labels)
print(labels_one_hot)
print(labels_one_hot.shape)

labels_smoothing = smooth_labels(labels_one_hot)
print(labels_smoothing)