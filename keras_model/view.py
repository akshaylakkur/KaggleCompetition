import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import zarr
import numpy as np

# Make this bigger to generate a dense grid.
N = 8

# Create some random data.
m = 3
x = 30
y = 30
z = 30

volume = zarr.load("data/train/static/ExperimentRuns/TS_5_4/VoxelSpacing10.000/ctfdeconvolved.zarr")[2][:x*m,:y*m,:z*m]
v2 = np.zeros((x,y,z))
for i in range(x):
    for j in range(y):
        for k in range(z):
            v2[i, j, k] = (volume[i:i+m, j:j+m, k:k+m].ravel()).mean()
volume = v2
rvl = volume.ravel()
m = min(rvl)
mx = max(rvl)
volume = (volume-m)/(mx-m)

# Create the x, y, and z coordinate arrays.  We use 
# numpy's broadcasting to do all the hard work for us.
# We could shorten this even more by using np.meshgrid.
x = np.arange(volume.shape[0])[:, None, None]
y = np.arange(volume.shape[1])[None, :, None]
z = np.arange(volume.shape[2])[None, None, :]
x, y, z = np.broadcast_arrays(x, y, z)

# Turn the volumetric data into an RGB array that's
# just grayscale.  There might be better ways to make
# ax.scatter happy.
c = np.tile(volume.ravel()[:, None], [1, 3])

# Do the plotting in a single call.
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(x.ravel(),
           y.ravel(),
           z.ravel(),
           c=c)
fig.show()
i = input()
