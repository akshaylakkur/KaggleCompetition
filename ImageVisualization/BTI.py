import numpy as np
import zarr
import matplotlib.pyplot as plt

#replace with your file - make sure it's denoised
filePath = '/Users/akshaylakkur/GitHub/KaggleCompetition/data/train/static/ExperimentRuns/TS_69_2/VoxelSpacing10.000/denoised.zarr'
zarrData = zarr.open(filePath, 'r')
zarrData = zarrData[0]



print(zarrData.shape)

#x/width image
plt.figure(figsize=(8, 8))
plt.imshow(zarrData[:, :, (zarrData.shape[0]//2 - 10)], cmap='gray')
plt.show()

#y/height image
plt.figure(figsize=(8, 8))
plt.imshow(zarrData[:, (zarrData.shape[0]//2), :], cmap='gray')
plt.show()

#z/depth image
plt.figure(figsize=(8, 8))
plt.imshow(zarrData[(zarrData.shape[0] // 2) , :, :], cmap='gray')
plt.show()