import napari
import numpy as np
import zarr

# Load your 3D data
#image_3d = np.random.rand(100, 100, 100)
imageNum = '5_4'
trainingFile = f'../data/train/static/ExperimentRuns/TS_{imageNum}/VoxelSpacing10.000/denoised.zarr'
image_3d = np.array(zarr.open(trainingFile)[0])
print(image_3d.shape)
sx, sy, sz = 90, 90, 90
subImage = image_3d[94:100,257:263,428:434]

# Visualize
viewer = napari.Viewer()
#viewer.add_image(image_3d, colormap="gray", scale=[1,1,1], opacity=1)
viewer.add_image(subImage, colormap="gray", scale=[1,1,1], opacity=1)
viewer.dims.ndisplay = 3
napari.run()


