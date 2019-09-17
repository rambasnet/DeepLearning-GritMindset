# https://www.tensorflow.org/tutorials/estimators/boosted_trees

from __future__ import absolute_import, division, print_function

import sys
import operator
import numpy as np
import pandas as pd
import tensorflow as tf
import csv
from tensorflow import keras
from sklearn.model_selection import StratifiedKFold
# tf.enable_eager_execution()

data_x = []
data_y = []

tf.enable_eager_execution()


def getData():
    global data_x
    global data_y
    inputFile = './data/GritMindset.csv'
    data_x = pd.read_csv(inputFile)
    # print([data[:]])
    label = 'HonorsScience'
    lblTypes = set(data_x[label])
    lblTypes = dict(zip(lblTypes, [0] * 2))
    lblTypes[2] = 1
    data[label] = data_x[label].map(lblTypes)
    data_y = data_x.pop('HonorsScience')


# Use entire batch since this is such a small dataset.
NUM_EXAMPLES = len(data_y)
