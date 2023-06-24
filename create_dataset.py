import pandas as pd
import scipy.io as scio
import matplotlib.pyplot as plt
import os
import numpy as np
import scipy
from tqdm import tqdm
import random

def read_diagnosis():
    diag_frame = pd.read_csv('diagnoses.csv')
    diag_dict = {}
    for line in diag_frame.values:
        diag_dict[line[0]] = line[1]
    return diag_dict


def get_diagnoses(file_path):
    f = open(file_path)
    file_data = f.readlines()
    diag_codes = []
    f.close()
    for line in file_data:
        if 'Dx' in line:
            diag_names_str = line.split(':')[1].split(',')
            for diagnosis_name in diag_names_str:
                diag_codes.append(int(diagnosis_name))
    return diag_codes


# diagnoses_dict = read_diagnosis()
# print(diagnoses_dict)

# df = pd.read_csv('./ptb_raw/hea/S0133.hea')
# diag_code = int(df.values[14][0].split(' ')[1])
# diagnosis = diagnoses_dict[diag_code]
# print(diagnosis)
# III V3 V5
# 1 2 3  avl avr avf v1 v2 v3 v4 v5 v6

# ecg_data_raw = scio.loadmat('./ptb_raw/mat/S0536.mat')['val']
# ecg_data = ecg_data_raw[2] # отведение III
# print(ecg_data_raw[[2, 8, 10]])
"""
plt.plot(ecg_data[:2500])
plt.show()
"""

b, a = scipy.signal.butter(2, [0.2, 100], btype='bandpass', fs=1000)

readed_diagnoses = {}
files_names_hea = os.listdir('./ptb_raw/hea')
files_names_mat = os.listdir('./ptb_raw/mat')
dataset_path = './dataset_mat/'

names_counter = 0
m_i_counter = 0  # количество экг с диагнозом инфаркт миакарда

processed_ecgs = []
labels = []
processed_ecgs_with_labels = []
#

for i in tqdm(range(len(files_names_hea))):
    diag_codes = get_diagnoses('./ptb_raw/hea/' + files_names_hea[i])
    if 164865005 in diag_codes:
        if m_i_counter > 40:  # количество нормальных ЭКГ 80, ограничиваем число диагноза 164865005, чтобы в датасет попали другие заболевания
            continue
        else:
            m_i_counter += 1
    ecg_data_raw = scio.loadmat('./ptb_raw/mat/' + files_names_mat[i])['val'][[2, 8, 10]]  # Отведения III, V3 и V5
    fragments_count = ecg_data_raw[0].size // 10000  # частота дискретизации 1000 Гц фрагменты по 10 с
    # label = 'normal' if get_diagnoses('./ptb_raw/hea/' + files_names_hea[i])[0] == 426783006 else 'abnormal'
    label = 0 if diag_codes[0] == 426783006 else 1
    for i in range(fragments_count):
        ecg_data = ecg_data_raw[:, i * 10000:(i + 1) * 10000]
        filtered = scipy.signal.filtfilt(b, a, ecg_data)[:, ::2]  # уменьшение частоты дискретизации в 2 раза
        processed_ecgs_with_labels.append([label, filtered])
        # np.savetxt(dataset_path + label + '_' + str(names_counter) + '.csv', filtered, delimiter=",")
        names_counter += 1

random.shuffle(processed_ecgs_with_labels)
for item in processed_ecgs_with_labels:
    labels.append(item[0])
    processed_ecgs.append(item[1])

test_data_count = len(processed_ecgs) // 5
scipy.io.savemat("train.mat", {"ecgs": processed_ecgs[test_data_count:], "labels": labels[test_data_count:]})
scipy.io.savemat("test.mat", {"ecgs": processed_ecgs[:test_data_count], "labels": labels[:test_data_count]})

# print(readed_diagnoses)
