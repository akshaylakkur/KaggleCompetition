import os
os.environ['KERAS_BACKEND'] = 'jax'

base_shape = (92, 315, 315)
data_in = base_shape
out_scale = 10
acc_scale = 0.05 # so that answer can be better predicted
scale = out_scale*acc_scale
data_out = tuple(int(dim*scale) for dim in base_shape)+(6,)

num_predict_per = 100

mdnum = "08"
