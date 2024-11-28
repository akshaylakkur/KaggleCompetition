import numpy as np
import os
import json
import pandas as pd
from dask.dataframe.shuffle import shuffle

picksDirectory = '/Users/akshaylakkur/GitHub/KaggleCompetition/data/train/overlay/ExperimentRuns/TS_69_2/Picks'

apoFerriten = []
thyroglubin = []
betaGalact = []
ribosome = []
vlp = []
betaAmylase = []


for file in os.listdir(picksDirectory):
    filePath = os.path.join(picksDirectory, file)

    'read files'
    with open(filePath, 'r') as f:
        data = json.load(f)
        protein = data.get('pickable_object_name')

        locations = [subPoint['location'] for subPoint in data.get('points')]

        if protein == "apo-ferritin":
            apoFerriten = locations
        elif protein == "thyroglobulin":
            thyroglubin = locations
        elif protein == "beta-galactosidase":
            betaGalact = locations
        elif protein == "ribosome":
            ribosome = locations
        elif protein == "virus-like-particle":
            vlp = locations
        else:
            betaAmylase = locations


particleDict = {
    "apoFerriten": apoFerriten,
    "thyroglubin": thyroglubin,
    "betaGalact": betaGalact,
    "ribosome": ribosome,
    "vlp": vlp,
}

counter = 0
for name, data in particleDict.items():
    df = pd.DataFrame(data)
    df['Class'] = counter
    df.to_csv(f'{name}.csv', index=False)
    counter += 1

apoFerriten = pd.read_csv('apoFerriten.csv')
betaGalact = pd.read_csv('betaGalact.csv')
ribosome = pd.read_csv('ribosome.csv')
vlp = pd.read_csv('vlp.csv')
thyroglubin = pd.read_csv('thyroglubin.csv')

FiveParticlesData = pd.concat([apoFerriten, betaGalact, ribosome, vlp, thyroglubin], )
FiveParticlesData = FiveParticlesData.sample(frac=1).reset_index(drop=True)
FiveParticlesData.to_csv('FiveParticlesData.csv', index=False)


