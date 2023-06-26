import os
import scipy.io as scio
from EcgRecognitionModel import EcgRecognitionModel
from EcgDataset import EcgDataset
import data_loader
import torch.nn as nn
import torch 

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


train_data, train_labels = data_loader.load_dataset('train.mat')
#test_data, test_labels = data_loader.load_dataset('test.mat')

train_dataset = torch.utils.data.DataLoader(EcgDataset(train_data, train_labels), 16)



device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = EcgRecognitionModel().to(device)

loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)

"""
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct/len(y_pred)) * 100
    return acc
"""
#y_pred = model(torch.Tensor(train_data[:3]))
#print(y_pred)
#print(torch.round(y_pred).squeeze())


#print(model(torch.Tensor(train_data[32:64]).to(device)).squeeze())


#train loop
for epoch in range(100):
    for batch in train_dataset:
        X, y = batch

        #print("labels = ", y)
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        y_pred = model(X)
        #print("pred = ", y_pred)
        loss = loss_fn(y_pred, y)
        #Обратное распротранение ошибки

        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch} loss is {loss.item()}")

with open('model_state_cnn.pt', 'wb') as f:
    torch.save(model.state_dict(), f)
