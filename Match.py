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
from scipy.spatial.distance import euclidean
#from soft_dtw_cuda.soft_dtw_cuda import SoftDTW

## Create the sequences
#batch_size, len_x, len_y, dims = 8, 15, 12, 5
#x = torch.rand((batch_size, len_x, dims), requires_grad=True)
#y = torch.rand((batch_size, len_y, dims))
## Transfer tensors to the GPU
#x = x.cuda()
#y = y.cuda()

## Create the "criterion" object
#sdtw = SoftDTW(use_cuda=True, gamma=0.1)

## Compute the loss value
#loss = sdtw(x, y)  # Just like any torch.nn.xyzLoss()

## Aggregate and call backward()
#loss.mean().backward()
#from Dtw import dtw as dtw
#import cupy as cp
#cp.cuda.Device(0).use()
			
class Match:

	def dtw_gpu(x, y):
    # Calculate the pairwise distance matrix using CuPy
		distance_matrix = cp.abs(cp.subtract.outer(x, y))

		# Initialize the DTW matrix with zeros
		dtw_matrix = cp.zeros((len(x), len(y)), dtype=cp.float32)

		# Fill the DTW matrix
		for i in range(len(x)):
			for j in range(len(y)):
				cost = distance_matrix[i, j]
				if i == 0 and j == 0:
					dtw_matrix[i, j] = cost
				elif i == 0:
					dtw_matrix[i, j] = cost + dtw_matrix[i, j - 1]
				elif j == 0:
					dtw_matrix[i, j] = cost + dtw_matrix[i - 1, j]
				else:
					dtw_matrix[i, j] = cost + min(dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1])

		# Return the DTW distance (bottom-right element of the matrix)
		return dtw_matrix[-1, -1]
	
	def fetch(ticker,bars=10,dt = None):
		
		tf = 'd'
		if dt != None:
			df = data(ticker,tf,dt,bars = bars+1)
		else:
			df = data(ticker,tf)
		df.load_np(bars,False)
		return df

	def worker(bar):
		df1, y = bar
		
		lis = []
		pbar = tqdm(total=len(df1.np))
		for x in df1.np:
			#print(f'-{x} , {y}-')
			x = x[0]
			distance = sfastdtw(x, y, 1, dist=euclidean)
			lis.append(distance)
			pbar.update(1)
		pbar.close()
		setattr(df1, 'index', x[1])
		setattr(df1,'scores',lis)
		return df1
	
	def match(ticker,dt,bars,dfs):
		y = Match.fetch(ticker,bars,dt).np[0][0]
		print(y)
		time.sleep(2)
		arglist = [[x,y] for x in dfs]
		dfs = main.pool(Match.worker,arglist)
		#df = [Match.worker(arg) for arg in arglist]
		return dfs
	
	def initiate(ticker, dt, bars): 
		ticker_list = screener.get('full')[:2000]
		dfs = main.pool(Match.fetch,ticker_list)
		start = datetime.datetime.now()
		dfs = Match.match(ticker,dt,bars,dfs)
		scores = []
		for df in dfs:
			lis = df.get_scores()
			scores += lis
		scores.sort(key=lambda x: x[2])
		print(f'completed in {datetime.datetime.now() - start}')
		return scores[:20]
		for ticker,index,score in scores[:20]:
			print(f'{ticker} {Data(ticker).df.index[index]}')
if __name__ == '__main__':
	
	if True:
		ticker_list = screener.get('full')[:100]
		dfs = main.pool(Match.fetch,ticker_list)
		ticker = 'JBL' #input('input ticker: ')
		dt = '2023-10-03' #input('input date: ')
		bars = 10 #int(input('input bars: '))
		start = datetime.datetime.now()
		dfs = Match.match(ticker,dt,bars,dfs)
		scores = []
		for df in dfs:
			t,i,s = df.get_scores()
			scores.append([t,i,s])
		print(scores)
		scores.sort(key=lambda x: x[1])
		print(f'completed in {datetime.datetime.now() - start}')
		for ticker,index,score in scores[:20]:
			
			print(f'{ticker} {data(ticker).df.index[index]} {score}')
		

						#lis.append(pyts.metrics.dtw(x,y))
				#lis.append(sax(x, y))
				#lis.append( dtw(x, y, method='sakoechiba', options={'window_size': 0.5}))
