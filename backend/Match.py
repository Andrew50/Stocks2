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


np_bars = 10

class Match:

    def load(tf):
        ticker_list = screener.get('full')[:100]
        df = pd.DataFrame({'ticker': ticker_list})
        df['dt'] = None
        df['tf'] = tf
        ds = Dataset(df)
        df = ds.load_np('dtw',np_bars)
        return df

    def run(ds, ticker, dt, tf):
        y = Data(ticker, tf, dt).load_np('dtw',np_bars)
        print(y)
        time.sleep(5)
        arglist = [[x, y, ticker, index] for index, x in ds]
        scores = Main.pool(Match.worker, arglist)
        scores.sort(key=lambda x: x[2])
        return scores[:20]

    def worker(bar):
        x, y, ticker, index = bar
        #print(f'[{x} {y}]')      
        distance = sfastdtw(x, y, 1, dist=euclidean)
        return [distance, ticker, index]


if __name__ == '__main__':

    ticker = 'JBL'  # input('input ticker: ')
    dt = '2023-10-03'  # input('input date: ')
    tf = 'd'  # int(input('input tf: '))
    start = datetime.datetime.now()
    ds = Match.load(tf)
    top_scores = Match.run(ds, ticker, dt, tf)

    for score, ticker, index in top_scores:

        print(f'{ticker} {Data(ticker).df.index[index]} {score}')
    print(f'completed in {datetime.datetime.now() - start}')
