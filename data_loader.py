import scipy.io as scio

import numpy as np

# 0 - нормальная ЭКГ, 1 - заболевание
def load_dataset(path_file):

    data = scio.loadmat(path_file)
    return data['ecgs'], data['labels']

