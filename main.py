import pandas as pd
import data_loader
import scipy
import numpy as np
import sys

def read_diagnosis():
    diag_frame = pd.read_csv('diagnoses.csv')
    diag_dict = {}
    for line in diag_frame.values:
        diag_dict[line[0]] = line[1]
    return diag_dict

np.set_printoptions(threshold=sys.maxsize)
data = scipy.io.loadmat('test.mat')
print(data['labels'])
