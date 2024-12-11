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

def data(istest=False):
    if istest:
        ls = test
    else:
        ls = train
    x = []
    y = []
    for exp in ls:
        for xi,yi in load(exp):
            x.append(xi)
            y.append(yi)
    return np.array(x), np.array(y)

def load(exp_no: str) -> (np.array, np.array):
    questions = [
        zarr.load(f'../data/train/static/ExperimentRuns/TS_{exp_no}/VoxelSpacing10.000/{typ}.zarr')[1]
        for typ in (
            'denoised',
            'ctfdeconvolved',
            'isonetcorrected',
            'wbp'
            )
        ]
    answer = []
    
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
        typ = []
        with open(f'../data/train/overlay/ExperimentRuns/TS_{exp_no}/Picks/{particle}.json') as f:
            dt = json.load(f)['points']
        for pt in dt:
            loc = pt['location']
            px, py, pz = loc['x'], loc['y'], loc['z']
            typ.append([px,py,pz])#+pad(p))
        typ.extend([
            [0,0,0]#+pad(p)
            for i in range(const.num_predict_per-len(typ))
            ])
        answer.extend(typ)
    for qn in questions:
        yield qn, answer

