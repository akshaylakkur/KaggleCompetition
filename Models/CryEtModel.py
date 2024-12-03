import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import zarr
import torch
from torch.nn import functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader, TensorDataset


'File setup'
imageNum = input('image number: ')
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
        self.coordinates = self.coordinates / 10
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
            image = self.transform(image, 30).expand(retrieveDataLength, -1, -1, -1, -1) #modify the size - just to test so that my comp. doesn't crash
        return dataset, image

'plot the 3d figure'
def plot3DFig(image, sliceIndex):
    'clean up tensor format'
    image = image.squeeze(0).squeeze(0).numpy()

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(projection='3d')

    # Extract the slice for plotting
    slice_2d = image[:, :, sliceIndex]  # Slicing along the first dimension (z-axis)

    # Create the x, y grid for the slice
    x = np.arange(slice_2d.shape[1])
    y = np.arange(slice_2d.shape[0])
    x, y = np.meshgrid(x, y)

    # Plot the 2D slice in 3D space
    ax.plot_surface(x, y, np.full_like(slice_2d, sliceIndex), rstride=1, cstride=1,
                    facecolors=plt.cm.viridis(slice_2d / np.max(slice_2d)),
                    shade=False,
                    cmap='grey')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"3D Plot of Slice {sliceIndex}")
    plt.show()

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
            nn.Linear(512, numClasses) # total classes
        )

        'regression function'
        self.boundingRegressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_size * 64, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 6) #6 coordinates for bounding box points
        )

    def forward(self, x, target_bboxes=None):
        'Extract Features'
        features = self.featureExtract(x)
        #print(f'features shape: {features.shape}')

        'Pass Through ROI Layer'
        proposals = self.RPN(features)
        #print(f'proposals shape: {proposals.shape}')

        'Through Adaptive Pooling'
        poolSolutions = self.ROIPooling(proposals)
        #print(f'pool solutions: {poolSolutions.shape}')

        poolSolutions = poolSolutions.view(poolSolutions.size(0), -1)
        #print(f'pool solutions after view: {poolSolutions.shape}')

        'Through Classification - for probability'
        classification = self.Classification(poolSolutions)
        #print(f'Classification Shape: {classification.shape}')

        'Through Regression - get bbox coords'
        bboxCoords = self.boundingRegressor(poolSolutions)
        #print(f'Bounding box shapes: {bboxCoords.shape}')

        if target_bboxes is not None:
            # During training, calculate losses: classification loss and bounding box regression loss
            classification_loss = F.cross_entropy(classification, target_bboxes['labels'])  # Use ground truth labels
            bbox_loss = F.smooth_l1_loss(bboxCoords, target_bboxes['bbox'])  # Compare with ground truth bounding box
            return classification, bboxCoords, classification_loss, bbox_loss
        else:
            return classification, bboxCoords

'prepare data'
dataset = CryEtDataset(trainingFile, coordsFile, transform=transform)
data, image = dataset.getitem()

TensorData = TensorDataset(data['bbox'], data['labels'])
dataloader = DataLoader(TensorData, batch_size=32, shuffle=False)

trueLabels = []
trueLabels.append(data['labels'])

'activate model'
model = CryEtModel(1, 32, 5)
testImage = torch.randn(size=(retrieveDataLength, 1, 5, 10, 10))

'set up loss & optimizers'
# classCriterion = nn.CrossEntropyLoss()
# regressiveLoss = nn.SmoothL1Loss()
Adam = torch.optim.Adam(model.parameters(), lr=0.01)
classLoss = []
boxLoss = []

predLabels = []
predBox = []

epochs = 20
for epoch in range(epochs):
    model.train()

    for batch in dataloader:
        boxCoords, classes = batch
        Adam.zero_grad()

        c, bc, cl, bl = model(testImage, target_bboxes={'labels' : data['labels'].squeeze(1), 'bbox' : data['bbox']})
        # cl = classCriterion(c, data['labels'])
        # bl = regressiveLoss(bc, data['bbox'])
        Loss = cl + bl
        Loss.backward()
        Adam.step()

        predClassLabels = torch.argmax(c, dim=1)
        predLabels.append(predClassLabels)
        predBox.append(bc)

        classLoss.append(cl.item())
        boxLoss.append(bl.item())
print(f'pred labels: {predLabels}')

print(f'class loss: {classLoss}')
print(f'box loss: {boxLoss}')

plt.figure()
plt.plot(classLoss)
plt.title('Class Loss')
plt.grid(True)
plt.show()

plt.figure()
plt.plot(boxLoss)
plt.title('Box Loss')
plt.grid(True)
plt.show()

