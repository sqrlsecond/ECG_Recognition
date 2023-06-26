import data_loader
import torch
from EcgRecognitionModel import EcgRecognitionModel
from EcgDataset import EcgDataset


test_data, test_labels = data_loader.load_dataset('test.mat')

device = 'cuda'

model = EcgRecognitionModel().to(device)
model.load_state_dict(torch.load('./model_state_cnn.pt'))
test_dataset = torch.utils.data.DataLoader(EcgDataset(test_data, test_labels), 16)

correct = 0
total = 0

with torch.no_grad():
    for batch in test_dataset:
        X, y = batch
        X, y = X.to(device), y.to(device)
        y_pred = torch.round(model(X))
        total += y.size(0)

        for i in torch.eq(y_pred, y):
            print(i)
            if i:
                correct += 1


print(correct)
print(total)
