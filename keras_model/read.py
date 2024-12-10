import zarr
import json
import numpy as np
import const

train = (
        '5_4',
        '69_2',
        '6_4',
        '6_6',
        '73_6',
        )
test = (
        '86_3',
        '99_9',
        )

def resize(arr):
    arr = np.array(arr)
    sh = arr.shape
    to_shape = ((0,const.num_predict-arr.shape[0]),(0,0))
    arr = np.pad(arr,to_shape)
    return arr

def data(istest=False):
    if istest:
        ls = test
    else:
        ls = train
    x = []
    y = []
    for exp in ls:
        xi, yi = load(exp)
        yi = resize(yi)
        x.append(xi)
        y.append(yi)
    return np.array(x), np.array(y)

def load(exp_no: str) -> (np.array, np.array):
    question = zarr.load(f'../data/train/static/ExperimentRuns/TS_{exp_no}/VoxelSpacing10.000/denoised.zarr')[1]
    answer = []
#    answer = np.zeros(const.data_out)
    
    floor = lambda x: int(x)
    ceil = lambda x: int(x+1)
    pad = lambda k: [0]*k+[1]+[0]*(5-k)
    for p,particle in enumerate((
            'apo-ferritin',
            'beta-amylase',
            'beta-galactosidase',
            'ribosome',
            'thyroglobulin',
            'virus-like-particle')):
        with open(f'../data/train/overlay/ExperimentRuns/TS_{exp_no}/Picks/{particle}.json') as f:
            dt = json.load(f)['points']
        for pt in dt:
            loc = pt['location']
            px, py, pz = loc['x'], loc['y'], loc['z']
            answer.append([px,py,pz]+pad(p))
#            px *= const.acc_scale
#            py *= const.acc_scale
#            pz *= const.acc_scale
#            for x in range(
#                    max(floor(px-5), const.data_out[0]),
#                    max(ceil(px+5), const.data_out[0])):
#                for y in range(
#                        max(floor(py-5), const.data_out[1]),
#                        max(ceil(py+5), const.data_out[1])):
#                    for z in range(
#                            max(floor(pz-5), const.data_out[2]),
#                            max(ceil(pz+5), const.data_out[2])):
#                        answer[z, x, y, p] = 1
    return question, answer

