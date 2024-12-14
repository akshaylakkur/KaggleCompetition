import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import zarr
import torch
from torch.nn import functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader, TensorDataset

retrieveDataLength = len(pd.read_csv(coordsFile)['Class'])

# import torch_xla
# import torch_xla.core.xla_model as xm
# import torch_xla.distributed.data_parallel as dp
# import torch_xla.distributed.xla_multiprocessing as xmp
'File setup'
imageNum = input('image number: ')

# trainingFile = f'/home/cde1/ml/akshay/ransModelRelated/DenoisedData/denoised_{imageNum}.zarr'
# coordsFile = f'/home/cde1/ml/akshay/ransModelRelated/SortedCoordsFiles/FiveParticlesDataTS_{imageNum}.csv'
trainingFile = f'/Users/akshaylakkur/GitHub/KaggleCompetition/data/train/static/ExperimentRuns/TS_{imageNum}/VoxelSpacing10.000/denoised.zarr'
coordsFile = f'/Users/akshaylakkur/PycharmProjects/KaggleComp/SortedCoordsFiles/FiveParticlesDataTS_{imageNum}.csv'

retrieveDataLength = len(pd.read_csv(coordsFile)['Class'])
class CryEtDataset(Dataset):
    def __init__(self, image, data, transform=None):
        'image setup'
        self.image = torch.from_numpy(np.array(zarr.open(image)[0])).unsqueeze(0).unsqueeze(0).float()
       
        self.data = pd.read_csv(data)
        self.transform = transform
        'extract info'
        self.coordinates = torch.tensor(self.data[['x', 'y', 'z']].values, dtype=torch.float32)
        #self.coordinates = self.coordinates / 10
        self.labels = torch.tensor(self.data[['Class']].values, dtype=torch.long)
        'create bbox'
        radiusDelta = 3
        self.bboxMin = self.coordinates - radiusDelta
        self.bboxMax = self.coordinates + radiusDelta

    def __len__(self):
        return len(self.data)

    def getshape(self):
        return self.image.shape

    def getitem(self):
        dataset = {
            'bbox' : torch.cat([self.bboxMin, self.bboxMax], dim=1),
            'labels' : self.labels
        }
        image = self.image
        if self.transform:
            image = self.transform(image, 20).expand(retrieveDataLength, -1, -1, -1, -1)
            #print(f'image shape: {image.shape}')
            return dataset, image

'transform shape to shrink it for better processing and training'
def transform(image, size, squeeze=False):
    shape = image.shape
    outputImage = F.interpolate(image, (shape[2] // size, shape[3] // size, shape[4] // size))
    #print(outputImage.shape)
    if squeeze:
        outputImage = outputImage.squeeze(0).squeeze(0).numpy()
    return outputImage

'Network Architecture'
class CryEtModel(nn.Module):
    def __init__(self, input_size, hidden_size, numClasses):
        super(CryEtModel, self).__init__()
        'Feature Extraction'
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
        'Pooling Function'
        self.ROIPooling = nn.AdaptiveMaxPool3d((4, 4, 4))

        'Classification Layer'
        self.Classification = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_size * 64, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, numClasses)  # total classes
        )

        'regression function'
        self.boundingRegressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_size * 64, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 6)  # 6 coordinates for bounding box points
        )

    def forward(self, x, target_bboxes=None):
        'Extract Features'
        features = self.featureExtract(x)
        # print(f'features shape: {features.shape}')

        'Pass Through ROI Layer'
        proposals = self.RPN(features)
        # print(f'proposals shape: {proposals.shape}')

        'Through Adaptive Pooling'
        poolSolutions = self.ROIPooling(proposals)
        # print(f'pool solutions: {poolSolutions.shape}')

        poolSolutions = poolSolutions.view(poolSolutions.size(0), -1)
        # print(f'pool solutions after view: {poolSolutions.shape}')

        'Through Classification - for probability'
        classification = self.Classification(poolSolutions)
        # print(f'Classification Shape: {classification.shape}')

        'Through Regression - get bbox coords'
        bboxCoords = self.boundingRegressor(poolSolutions)
        # print(f'Bounding box shapes: {bboxCoords.shape}')

        if target_bboxes is not None:
            classification_loss = classCriterion(classification, target_bboxes['labels'])  # Use ground truth labels
            bbox_loss = regressiveLoss(bboxCoords, target_bboxes['bbox'])  # Compare with ground truth bounding box
            Loss = classification_loss + bbox_loss
            Loss.backward()
            Adam.step()

            return classification, bboxCoords, Loss.item()
        else:
            return classification, bboxCoords

'prepare data'
dataset = CryEtDataset(trainingFile, coordsFile, transform=transform)
data, image = dataset.getitem()
print(f'image shape: {image.shape}')

TensorData = TensorDataset(data['bbox'], data['labels'])
dataloader = DataLoader(TensorData, batch_size=32, shuffle=False)

trueLabels = []
trueLabels.append(data['labels'])
print([i for i in data['labels']], end="")
print(f"true bbox coords: {data['bbox']}")


'activate model'

model = CryEtModel(1, 64, 5)#.to(device)

testImage = torch.randn(size=(retrieveDataLength, 1, 5, 10, 10))

'set up loss & optimizers'
classCriterion = nn.CrossEntropyLoss()#.to(device)
regressiveLoss = nn.SmoothL1Loss()#.to(device)
Adam = torch.optim.Adam(model.parameters(), lr=0.01)
Loss = []

predLabels = []
predBox = []
predLabelsFull = []


epochs = 1001
for epoch in range(epochs):
    model.train()

    for batch in dataloader:
        boxCoords, classes = batch
        Adam.zero_grad()

        c, bc, loss = model(image, target_bboxes={'labels': data['labels'].squeeze(1), 'bbox': data['bbox']})
        # c, bc = model(image.to(device), target_bboxes={'labels' : data['labels'].squeeze(1).to(device), 'bbox' : data['bbox'].to(device)})
        # cl = classCriterion(c, data['labels'])
        # bl = regressiveLoss(bc, data['bbox'])
        # Loss = cl + bl
        # Loss.backward()
        # Adam.step()
        #xm.optimizer_step(Adam)
        c = torch.softmax(c, dim=1)
        predLabelsFull.append(c)
        predClassLabels = torch.argmax(c, dim=1)
        predLabels.append(predClassLabels)
        predBox.append(bc)
        Loss.append(loss)

    if epoch % 250 == 0:
        print(f'epoch: {epoch}')
        print(f'Length of Subset: {len(predLabelsFull[-1])}')
        print(f'All 6 points for classification: {predLabelsFull[-1]}')
        print(f'pred labels: {predLabels[-1]}')
        print(f'pred coordinates: {predBox[-1]}')
        print(f'total loss: {Loss[-1]}')

#plt.figure()
#plt.plot(Loss)
#plt.title('Class Loss')
#plt.grid(True)
#plt.show()







