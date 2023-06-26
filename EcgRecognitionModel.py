import torch
import torch.nn as nn

class EcgRecognitionModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        #3 отведения ЭКГ, размер фильтра 5, 128 фильтров
        #5000 на входе, 4996 на выходе
        self.cnn1 = nn.Conv1d(3, 128, 5)
        self.bn1 = nn.BatchNorm1d(128)
        self.act1 = nn.ReLU()
        #2498 на выходе
        self.mp1 = nn.MaxPool1d(2, stride=2)

        #2 блок
        # 2494 на выходе
        self.cnn2 = nn.Conv1d(128, 256, 5)
        self.bn2 = nn.BatchNorm1d(256)
        self.act2 = nn.ReLU()
        # 1247 на выходе
        self.mp2 = nn.MaxPool1d(2, stride=2)


        #3 блок
        # 2494 на выходе
        self.cnn3 = nn.Conv1d(256, 128, 5)
        self.bn3 = nn.BatchNorm1d(128)
        self.act3 = nn.ReLU()
        # 1247 на выходе
        self.mp3 = nn.MaxPool1d(2, stride=2)

        #4 блок
        self.fl4 = nn.Flatten()
        #1247 * 256 = 319 232
        self.ln4_1 = nn.Linear(79488, 512)
        self.act4_1 = nn.ReLU()
        self.ln4_2 = nn.Linear(512, 1)# Бинарная классификация
        self.act4_2 = nn.Sigmoid()

        self.ln_temp = nn.Linear(159872, 1)

        """
        self.fl1 = nn.Flatten()
        self.ln1 = nn.Linear(3*5000, 512)
        self.act1 = nn.ReLU()
        self.ln2 = nn.Linear(512, 1)
        self.act2 = nn.Sigmoid()
        """

    def forward(self, x):

        #1 блок
        x = self.cnn1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.mp1(x)

        x = self.cnn2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.mp2(x)

        x = self.cnn3(x)
        x = self.bn3(x)
        x = self.act3(x)
        x = self.mp3(x)

        x = self.fl4(x)
        x = self.ln4_1(x)
        x = self.act4_1(x)
        x = self.ln4_2(x)
        x = self.act4_2(x).squeeze()

        """
        x = self.fl1(x)
        x = self.act1(self.ln1(x))
        x = self.act2(self.ln2(x))
        """
        return x
