from locale import normalize
from multiprocessing.pool import Pool
from Data import Data as data
from Data import Main as main
import numpy as np
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



if __name__ == '__main__':
    df = data('AAPL', 'd')
    returns = df.load_np(10, False)
    print(returns[1])
    