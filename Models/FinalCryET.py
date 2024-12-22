''' Input should be a concatenated tensor consisting of image, respective classes, and bounding box coordinates '''
''' Output consists of class and bounding box coordinates '''

import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
import gc
import zarr
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
#import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, TensorDataset


'training file local'
#imageNum = input('image number: ')
trainingFile = f'/Users/akshaylakkur/GitHub/KaggleCompetition/data/train/static/ExperimentRuns/TS_5_4/VoxelSpacing10.000/denoised.zarr'
coordsFile = f'/Users/akshaylakkur/PycharmProjects/KaggleComp/SortedCoordsFiles/FiveParticlesDataTS_5_4.csv'

'training file Kaggle'
# trainingFile = f'/kaggle/input/czii-cryo-et-object-identification/train/static/ExperimentRuns/TS_{imageNum}/VoxelSpacing10.000/denoised.zarr'
# coordsFile = f'/kaggle/input/cryetcoords-and-classfilesorted/SortedCoordsFiles/FiveParticlesDataTS_{imageNum}.csv'

'training file gpu'
# trainingFile = f'/home/cde1/ml/akshay/ransModelRelated/DenoisedData/denoised_{imageNum}.zarr'
# coordsFile = f'/home/cde1/ml/akshay/ransModelRelated/SortedCoordsFiles/FiveParticlesDataTS_{imageNum}.csv'

'Set up Data'
data = pd.read_csv(coordsFile)
dataLength = len(data['x'])
class CryEtDataset(Dataset):
    def __init__(self, image, data, transform=None):
        self.image = torch.from_numpy(np.array(zarr.open(image)[0])).unsqueeze(0).unsqueeze(0).expand(dataLength, -1,-1,-1,-1).float()
        'extract coordinates'
        self.coordinates = torch.tensor(data[['x','y','z']].values/10, dtype=torch.int)
        self.labels = torch.tensor(data[['Class']].values, dtype=torch.long)
        self.transform = transform

        'setup bounding boxes'
        radiusDelta = 2
        self.bboxMax = self.coordinates + 2
        self.bboxMin = self.coordinates - 2

    def ToDataset(self):
        image = self.image
        orderedIndexes = [0, 3, 1, 4, 2, 5]
        dataset = {
            'bbox' : torch.cat([self.bboxMin, self.bboxMax], dim=1)[:, orderedIndexes],
            'labels' : self.labels
        }
        return dataset, image


class CryEtModel(nn.Module):
    def __init__(self, input_size, hidden_size, numClasses):
        super(CryEtModel, self).__init__()
        'feature extractions'
        self.featureExtract = nn.Sequential(
            nn.Conv3d(input_size, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=1),
        )

        'ROI Layer'
        self.RPN = nn.Sequential(
            nn.Conv3d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=1),
        )

        self.classification = nn.Sequential(
            nn.Conv3d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(hidden_size*48, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, numClasses)
        )

        self.boundingRegression = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_size*48, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, 6),
        )

    def forward(self, image):
        features = self.featureExtract(image)
        print(features.shape)
        proposals = self.RPN(features)
        print(proposals.shape)
        classification = self.classification(proposals)
        print(classification.shape)
        bboxes = self.boundingRegression(proposals)
        return classification, bboxes

dataset = CryEtDataset(trainingFile, data, transform=None)
dataset, image = dataset.ToDataset()

test = torch.randn(size=(130, 1, 4,4,4))
model = CryEtModel(1, 5, 5)
y = model(test)
'set up dataset'
dataset = TensorDataset(dataset['bbox'], dataset['labels'])
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for item in dataloader:
    for x in item[0]:
        print(image[:, :, int(x[4]):int(x[5]), int(x[2]):int(x[3]), int(x[0]):int(x[1])].shape)
        break


'set up losses and optimizers along with training lists'
classificationCriterion = nn.CrossEntropyLoss()
regressionCriterion = nn.MSELoss()
Adam = optim.Adam(model.parameters(), lr=0.001)
epochs = 1

predLabels = []
predBoxes = []
lossVals = []

for epoch in range(epochs):
    model.train()
    Adam.zero_grad()
    for item in dataloader:
        for x in item[0]:
            c, bb = model(image[:, :, int(x[4]):int(x[5]), int(x[2]):int(x[3]), int(x[0]):int(x[1])])
            classLoss = classificationCriterion(c, item[1])
            regressionLoss = regressionCriterion(bb, item[0])
            Loss = classLoss + regressionLoss
            Loss.backward()
            Adam.step()
            labels = torch.argmax(c, dim=1)

            'read the outputs'
            predLabels.append(labels)
            predBoxes.append(bb)
            lossVals.append(Loss.item())

    if epoch % 25 == 0:
        print(f'Epoch {epoch}')
        print(f'Pred Labels: {predLabels[-1]}')
        print(f'Pred Boxes: {predBoxes[-1]}')



    torch.cuda.empty_cache()
    gc.collect()

plt.figure()
plt.plot(lossVals)
plt.title('Loss Values')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

torch.save(model.state_dict(), 'CryEtModel.pth')
