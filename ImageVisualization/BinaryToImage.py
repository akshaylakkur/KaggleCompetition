import zarr
import numpy as np
import napari as nap
import matplotlib.pyplot as plt

filePath = '/Users/akshaylakkur/GitHub/KaggleCompetition/data/train/static/ExperimentRuns/TS_69_2/VoxelSpacing10.000/denoised.zarr'
#filePath = '/Users/akshaylakkur/GitHub/KaggleCompetition/data/train/static/ExperimentRuns/TS_5_4/VoxelSpacing10.000/denoised.zarr'

toZarrFile = zarr.open(filePath, 'r')

highestRes = toZarrFile[0]

toNumArr = np.array(highestRes)

to3d = nap.Viewer()
to3d.add_image(toNumArr, colormap='gray', name="3D Tomogram")
nap.run()


