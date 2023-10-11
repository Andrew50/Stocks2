from bisect import insort_right
import numpy as np
import websocket, datetime, os, pyarrow, shutil,statistics, warnings, math, time, pytz, tensorflow, random
from pyarrow import feather
import asyncio
from multiprocessing import Pool
import pandas as pd
import os, time 
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import numpy
from sklearn import preprocessing
import pandas as pd
import mplfinance as mpf
import PySimpleGUI as sg
import matplotlib.ticker as mticker
from matplotlib import pyplot as plt
from multiprocessing.pool import Pool
import os, pathlib, shutil, math, PIL, io
from tensorflow.keras import models
#models import load_model
## Implement error getting passed into discord 




class Main:
	def pool(deff,arg):
		pool = Pool()
		data = list(tqdm(pool.imap_unordered(deff, arg), total=len(arg)))
		return data
	def is_pre_market(dt):
		if dt is None: return False
		if( (((dt.hour*60)+dt.minute) - 570) < 0):
			return True
		return False
	def format_date(dt):
		if dt is None: return None
		if dt == 'current': return datetime.datetime.now(pytz.timezone('EST'))
		if isinstance(dt,str):
			try: dt = datetime.datetime.strptime(dt, '%Y-%m-%d')
			except: dt = datetime.datetime.strptime(dt, '%Y-%m-%d %H:%M:%S')
		time = datetime.time(dt.hour,dt.minute,0)
		dt = datetime.datetime.combine(dt.date(),time)
		if dt.hour == 0 and dt.minute == 0:
			time = datetime.time(9,30,0)
			dt = datetime.datetime.combine(dt.date(),time)
		return dt	
	def data_path(ticker,tf):
		if 'd' in tf or 'w' in tf: path = 'd/' 
		else: path = '1min/'
		return 'local/data/' + path + ticker + '.feather'
	
	def is_market_open(): # Change to a boolean at some point 
		if(datetime.datetime.now().weekday() >= 5): return False # If saturday or sunday 
		dt = datetime.datetime.now(pytz.timezone('US/Eastern'))
		hour = dt.hour
		minute = dt.minute
		if hour >= 10 and hour <= 16: return True
		elif hour == 9 and minute >= 30: return True
		return False
	

	def worker(path):  ###worker for the main fucntion. this will be multiprcoessed by main.
		try: 
			with open(path, 'rb') as file: return np.load(file,allow_pickle=True)
		except: pass
	def fast_get(tickers,tf = 'd'): ##this is for when you just want to pull the df for example when doing ml 
		arglist = [f'local/data/{tf}/{ticker}.npy' for ticker in tickers]
		dfs = Main.pool(Main.worker,arglist)
		return dfs
	
	def load_model(st):
		start = datetime.datetime.now()
		print(st)
		model = models.load_model('C:/Stocks/sync/models/model_' + st)
		print(f'{st} model loaded in {datetime.datetime.now() - start}')
		return model

	def score_worker(bar):
		
		dfs, st, threshold = bar
		model = Main.load_model(st)
		returns = []
		for df in dfs:
			scores = df.load_score(st,threshold)
			for bar in [b for b in scores if b[3] > threshold]:
				returns += bar
		return returns
		
	def worker2(bar):
		ticker, dt, tf = bar
		start = datetime.datetime.now()
		bars = 50
		if dt == None: bars = 0
		df = Data(ticker,tf,dt,bars = bars)
		tim = (datetime.datetime.now() - start)
		start = datetime.datetime.now()
		
		df.np = df.load_np(bars)
		return df
	
	
		
	def score_dataset(df,sts,threshold = None):
		arglist = [df.iloc[i] for i in range(len(df))]
		dfs = Main.pool(Main.worker2,arglist)
		if threshold == None:
			threshold = Main.get_config('Screener threshold')
		arglist = [[dfs,st,threshold] for st in sts]
		setups = Main.pool(Main.score_worker,arglist)
		return setups
	
	def get_config(name):
		s  = open("C:/Stocks/config.txt", "r").read()
		trait = name.split(' ')[1]
		script = name.split(' ')[0]
		trait.replace(' ','')
		bars = s.split('-')
		found = False
		for bar in bars:
			if script in bar: 
				found = True
				break
		if not found: raise Exception(str(f'{script} not found in config'))
		lines = bar.splitlines()
		found = False
		for line in lines:
			if trait in line.split('=')[0]: 
				found = True
				break
		if not found: raise Exception(str(f'{trait} not found in config'))
		value = line.split('=')[1].replace(' ','')
		try: value = float(value)
		except: pass
		return value
		
		


class Data:
	
	def __init__(self,ticker = 'QQQ',tf = 'd',dt = None,bars = 0,offset = 0,value = None, pm = True):
		try:
			if len(tf) == 1: tf = '1' + tf
			dt = Main.format_date(dt)
			if 'd' in tf or 'w' in tf: base_tf = '1d'
			else: base_tf = '1min'
			try: df = feather.read_feather(Main.data_path(ticker,tf)).set_index('datetime',drop = True)
			except FileNotFoundError: raise TimeoutError
			if df.empty: raise TimeoutError
			if pm: 
				pm_bar = pd.read_feather('C:/Stocks/sync/files/current_scan.feather').set_index('ticker').loc[ticker]
				pm_price = pm_bar['pm change'] + df.iat[-1,3]
				df = pd.concat([df,pd.DataFrame({'datetime': [datetime.datetime.now()], 'open': [pm_price],'high': [pm_price], 'low': [pm_price], 'close': [pm_price], 'volume': [pm_bar['pm volume']]}).set_index("datetime",drop = True)])
			if dt != None:
				try: df = df[:Data.findex(df,dt) + 1 + int(offset*(pd.Timedelta(tf) / pd.Timedelta(base_tf)))]
				except IndexError: raise TimeoutError
			if tf != '1min' or not pm: df = df.between_time('09:30', '15:59')
			if 'w' in tf:
				last_bar = df.tail(1)
				df = df[:-1]
			df = df.resample(tf,closed = 'left',label = 'left',origin = pd.Timestamp('2008-01-07 09:30:00')).apply({'open':'first','high':'max','low':'min','close':'last','volume':'sum'})
			if 'w' in tf: df = pd.concat([df,last_bar])
			df = df.dropna()[-bars:]
		except TimeoutError: df = pd.DataFrame()
		self.df = df
		self.len = len(df)
		self.ticker = ticker
		self.tf = tf
		self.dt = dt
		self.value = value
		self.bars = bars
		self.offset = offset
		
	def load_np(self,bars,standard = True,gpu = False):
		returns = []
		try:
			
			df = self.df
			if len(df) < 5: return returns
			if standard: partitions = 1
			else: partitions = bars//2
			x = df.to_numpy()
			x = np.flip(x,0)
			
			d = np.zeros((x.shape[0]-1,x.shape[1]))
			for i in range(len(d)): #add ohlc
				d[i] = x[i+1]/x[i,3] - 1
			if partitions != 0:
				for i in list(range(bars,d.shape[0]+1,partitions)):
					if gpu:
						#x = x.reshape(1, 2, bars)
						#x = torch.tensor(list(x), requires_grad=True).cuda()
						#sequence2 = torch.tensor([1.0, 2.0, 2.5, 3.5], requires_grad=True).cuda()
						pass
					else:
						x = d[i-bars:i]	
						x = preprocessing.normalize(x,axis = 0)
						if not standard: 
							x = x[:,3]
							x = np.column_stack((x, numpy.arange(  x.shape[0])))
						returns.append([x,i])
		except TimeoutError: 
			pass
		return returns
	
	def load_plot(self,hidden = False):
		buffer = io.BytesIO()
		s = mpf.make_mpf_style(base_mpf_style= 'nightclouds',marketcolors=mpf.make_marketcolors(up='g',down='r',wick ='inherit',edge='inherit',volume='inherit'))
		if hidden: title = ''
		else: title = f'{self.ticker}  {self.dt}  {self.tf}  {self.score}'
		if self.offset == 0: _, axlist = mpf.plot(self.df, type='candle', axisoff=True,volume=True, style=s, returnfig = True, title = title, figratio = (Main.get_config('Study chart_aspect_ratio'),1),figscale=Main.get_config('Study chart_size'), panel_ratios = (5,1),  tight_layout = True,vlines=dict(vlines=[self.dt], alpha = .25))
		else: _, axlist =  mpf.plot(df, type='candle', volume=True,axisoff=True,style=s, returnfig = True, title = title, figratio = (Main.get_config('Study chart_aspect_ratio'),1),figscale=Main.get_config('Study chart_size'), panel_ratios = (5,1),  tight_layout = True)
		ax = axlist[0]
		ax.set_yscale('log')
		ax.yaxis.set_minor_formatter(mticker.ScalarFormatter())
		plt.savefig(buffer, format='png')
		self.plot = buffer.getvalue()
		return self.plot

	def load_score(self,st,model = None):
		if model == None: model = Main.load_model(st)
		returns = []
		for df, index in self.np:
			score = model.predict(df)
			returns.append([self.ticker,self.df.index[index],st,score])
		self.score = returns
		return returns

	def findex(self,dt):
		dt = Main.format_date(dt)
		if not isinstance(self,pd.DataFrame): df = self.df
		else: df = self
		i = int(len(df)/2)
		k = int(i/2)
		while k != 0:
			date = df.index[i].to_pydatetime()
			if date > dt: i -= k
			elif date < dt: i += k
			k = int(k/2)
		while df.index[i].to_pydatetime() < dt: i += 1
		while df.index[i].to_pydatetime() > dt: i -= 1
		return i
		




if __name__ == '__main__':
	df = Data(dt = '2023-10-06')
	print(df.df)
	






























