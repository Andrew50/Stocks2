
import numpy as np
import os, time
import pandas as pd

path1 = 'C:/Stocks/local/data/d/'
path2 = 'C:/Stocks2/local/data/d/'
items = os.listdir(path1)


for path in items:
    df = pd.read_feather(path1 + path)
    print(df)
    df = df.to_numpy
    print(df)
    time.sleep(10000)
   # np.save(path2 + path[:-7] + 'npy', df)







