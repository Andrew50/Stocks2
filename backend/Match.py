from locale import normalize
from multiprocessing.pool import Pool
from Data import Main, Data, Dataset
import numpy as np
import pandas as pd
import datetime
from Screener import Screener as screener
import time
from discordwebhook import Discord
import numpy as np
from sklearn import preprocessing
from sfastdtw import sfastdtw
import mplfinance as mpf
import torch
from tqdm import tqdm
from sfastdtw import sfastdtw
from scipy.spatial.distance import euclidean
# from soft_dtw_cuda.soft_dtw_cuda import SoftDTW

# Create the sequences
# batch_size, len_x, len_y, dims = 8, 15, 12, 5
# x = torch.rand((batch_size, len_x, dims), requires_grad=True)
# y = torch.rand((batch_size, len_y, dims))
# Transfer tensors to the GPU
# x = x.cuda()
# y = y.cuda()

# Create the "criterion" object
# sdtw = SoftDTW(use_cuda=True, gamma=0.1)

# Compute the loss value
# loss = sdtw(x, y)  # Just like any torch.nn.xyzLoss()

# Aggregate and call backward()
# loss.mean().backward()
# from Dtw import dtw as dtw
# import cupy as cp
# cp.cuda.Device(0).use()


class Match:

    def load(tf):
        ticker_list = screener.get('full')
        df = pd.DataFrame({'ticker': ticker_list})
        df['dt'] = None
        df['tf'] = tf
        print('1')
        ds = Dataset(df)
        print('2')
        df = ds.load_np()
        return df

    def run(ds, ticker, dt, tf):
        y = Data(ticker, tf, dt).load_np()[0]
        arglist = [[x, y, ticker, index] for index, x in ds]
        scores = Main.pool(Match.worker, arglist)
        scores.sort(key=lambda x: x[2])
        return scores[:20]

    def worker(bar):
        x, y, ticker, index = bar
        distance = sfastdtw(x, y, 1, dist=euclidean)
        return [distance, ticker, index]


if __name__ == '__name__':

    ticker = 'JBL'  # input('input ticker: ')
    dt = '2023-10-03'  # input('input date: ')
    tf = 'd'  # int(input('input tf: '))
    start = datetime.datetime.now()
    ds = Match.load(tf)
    top_scores = Match.run(ds, ticker, dt, tf)

    for score, ticker, index in top_scores:

        print(f'{ticker} {Data(ticker).df.index[index]} {score}')
    print(f'completed in {datetime.datetime.now() - start}')
    # lis.append(pyts.metrics.dtw(x,y))
    # lis.append(sax(x, y))
    # lis.append( dtw(x, y, method='sakoechiba', options={'window_size': 0.5}))
