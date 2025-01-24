import const
import pandas as pd

out = [['1','2','3','4','5']] # model output
df = pd.DataFrame
df=df(out,columns=([
    'experiment',
    'particle_type',
    'x',
    'y',
    'z',
]))
df.to_csv("submission.csv")
