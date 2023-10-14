from tensorflow.keras.models import Sequential
from tensorflow.keras import models
import io
import PIL
import pathlib
from multiprocessing.pool import Pool
from bisect import insort_right
import numpy as np
import websocket
import datetime
import os
import pyarrow
import shutil
import statistics
import warnings
import math
import time
import pytz
import tensorflow
import random
from pyarrow import feather
import asyncio
from multiprocessing import Pool, current_process
import pandas as pd
import os
import time
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import numpy
from sklearn import preprocessing
import pandas as pd
import mplfinance as mpf
import PySimpleGUI as sg
import matplotlib.ticker as mticker
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm
from pyarrow import feather
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, load_model
from multiprocessing import Pool, current_process
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Dropout
import websocket
import datetime
import os
import pyarrow
import shutil
import statistics
import warnings
import math
import time
import pytz
import tensorflow
import random
warnings.filterwarnings("ignore")


# models import load_model


	
class Main:
	
	
	

	def sample(st, use):
		Main.consolidate_database()
		allsetups = pd.read_feather('local/data/' + st + '.feather').sort_values(
			by='dt', ascending=False).reset_index(drop=True)
		yes = []
		no = []
		groups = allsetups.groupby(pd.Grouper(key='ticker'))
		dfs = [group for _, group in groups]
		for df in dfs:
			df = df.reset_index(drop=True)
			for i in range(len(df)):
				bar = df.iloc[i]
				if bar['value'] == 1:
					for ii in [i + ii for ii in [-2, -1, 1, 2]]:
						if abs(ii) < len(df):
							bar2 = df.iloc[ii]
							if bar2['value'] == 0:
								no.append(bar2)
					yes.append(bar)
		yes = pd.DataFrame(yes)
		no = pd.DataFrame(no)
		required = int(len(yes) - ((len(no)+len(yes)) * use))
		if required < 0:
			no = no[:required]
		while True:
			no = no.drop_duplicates(subset=['ticker', 'dt'])
			required = int(len(yes) - ((len(no)+len(yes)) * use))
			sample = allsetups[allsetups['value'] == 0].sample(frac=1)
			if required < 0 or len(sample) == len(no):
				break
			sample = sample[:required + 1]
			no = pd.concat([no, sample])
		df = pd.concat([yes, no]).sample(frac=1).reset_index(drop=True)
		df['tf'] = st.split('_')[0]
		df = df[['ticker', 'dt', 'tf']]
		return df

	def mass_train(st, use, epochs):
		df = sample(st, use)
		sample = Dataset(df)

	def pool(deff, arg):
		return list(tqdm(Pool().imap_unordered(deff, arg), total=len(arg)))
	
	def is_pre_market(dt):
		if dt != None and ((((dt.hour*60)+dt.minute) - 570) < 0): return True
		return False

	def format_date(dt):
		if dt is None: return None
		if dt == 'current': return datetime.datetime.now(pytz.timezone('EST'))
		if isinstance(dt, str):
			try: dt = datetime.datetime.strptime(dt, '%Y-%m-%d')
			except: dt = datetime.datetime.strptime(dt, '%Y-%m-%d %H:%M:%S')
		time = datetime.time(dt.hour, dt.minute, 0)
		dt = datetime.datetime.combine(dt.date(), time)
		if dt.hour == 0 and dt.minute == 0:
			time = datetime.time(9, 30, 0)
			dt = datetime.datetime.combine(dt.date(), time)
		return dt

	def data_path(ticker, tf):
		if 'd' in tf or 'w' in tf: path = 'd/'
		else: path = '1min/'
		return 'local/data/' + path + ticker + '.feather'

	def train(st, percent_yes, epochs):
		df = pd.read_feather('local/data/' + st + '.feather')
		ones = len(df[df['value'] == 1])
		if ones < 150:
			print(f'{st} cannot be trained with only {ones} positives')
			return
		df = Main.sample(st, percent_yes)
		print(df)
		arglist = [df.iloc[i] for i in range(len(df))]
		dfs = Main.pool(Main.worker2, arglist)

		model = Sequential([Bidirectional(LSTM(64, input_shape=(x.shape[1], x.shape[2]), return_sequences=True,),), Dropout(
			0.2), Bidirectional(LSTM(32)), Dense(3, activation='softmax'),])
		model.compile(loss='sparse_categorical_crossentropy',
					  optimizer=Adam(learning_rate=1e-3), metrics=['accuracy'])
		model.fit(x, y, epochs=epochs, batch_size=64, validation_split=.2,)
		model.save('sync/models/model_' + st)
		tensorflow.keras.backend.clear_session()

	def is_market_open():  # Change to a boolean at some point
		if (datetime.datetime.now().weekday() >= 5):
			return False  # If saturday or sunday
		dt = datetime.datetime.now(pytz.timezone('US/Eastern'))
		hour = dt.hour
		minute = dt.minute
		if hour >= 10 and hour <= 16:
			return True
		elif hour == 9 and minute >= 30:
			return True
		return False

	# worker for the main fucntion. this will be multiprcoessed by main.
	def worker(path):
		try:
			with open(path, 'rb') as file:
				return np.load(file, allow_pickle=True)
		except:
			pass

	# this is for when you just want to pull the df for example when doing ml
	def fast_get(tickers, tf='d'):
		arglist = [f'local/data/{tf}/{ticker}.npy' for ticker in tickers]
		dfs = Main.pool(Main.worker, arglist)
		return dfs

	def load_model(st):
		start = datetime.datetime.now()
		model = models.load_model('sync/models/model_' + st)
		print(f'{st} model loaded in {datetime.datetime.now() - start}')
		return model

	def score_worker(bar):

		dfs, st, threshold = bar
		model = Main.load_model(st)
		returns = []
		for df in dfs:
			scores = df.load_score(st, threshold)
			for bar in [b for b in scores if b[3] > threshold]:
				returns += bar
		return returns

	def worker2(bar):
		ticker, dt, tf = bar
		start = datetime.datetime.now()
		bars = 50
		if dt == None:
			bars = 0
		df = Data(ticker, tf, dt, bars=bars)
		tim = (datetime.datetime.now() - start)
		start = datetime.datetime.now()

		df.np = df.load_np(bars)
		return df

	def score_dataset(df, sts, threshold=None):
		arglist = [df.iloc[i] for i in range(len(df))]
		dfs = Main.pool(Main.worker2, arglist)
		if threshold == None:
			threshold = Main.get_config('Screener threshold')
		arglist = [[dfs, st, threshold] for st in sts]
		setups = Main.pool(Main.score_worker, arglist)
		return setups

	def get_config(name):
		s = open("config.txt", "r").read()
		trait = name.split(' ')[1]
		script = name.split(' ')[0]
		trait.replace(' ', '')
		bars = s.split('-')
		found = False
		for bar in bars:
			if script in bar:
				found = True
				break
		if not found:
			raise Exception(str(f'{script} not found in config'))
		lines = bar.splitlines()
		found = False
		for line in lines:
			if trait in line.split('=')[0]:
				found = True
				break
		if not found:
			raise Exception(str(f'{trait} not found in config'))
		value = line.split('=')[1].replace(' ', '')
		try:
			value = float(value)
		except:
			pass
		return value



	def run():
		Main.check_directories()
		# current_day = Data.format_date(yf.download(tickers = 'QQQ', period = '25y', group_by='ticker', interval = '1d', ignore_tz = True, progress = False, show_errors = False, threads = False, prepost = False).index[-1-Data.is_market_open()])
		# current_minute = Data.format_date(yf.download(tickers = 'QQQ', period = '5d', group_by='ticker', interval = '1m', ignore_tz = True, progress = False, show_errors = False, threads = False, prepost = False).index[-1-Data.is_market_open()])
		from Screener import Screener as screener
		scan = screener.get('full', True)
		batches = []
		# for i in range(len(scan)):
		#   ticker = scan[i]
		#   batches.append([ticker, current_day, 'd'])
		#   batches.append([ticker, current_minute, '1min'])
		for i in range(len(scan)):
			for tf in ['d', '1min']:
				batches.append([scan[i], tf])
		Data.pool(Data.update, batches)
		if Data.get_config("Data identity") == 'laptop':
			weekday = datetime.datetime.now().weekday()
			if weekday == 4:
				Data.backup()
			elif weekday == 5:
				Data.retrain_models()
		Data.refill_backtest()

	def check_directories():
		dirs = ['local', 'local/data', 'local/account', 'local/study',
				'local/trainer', 'local/data/1min', 'local/data/d']
		if not os.path.exists(dirs[0]):
			for d in dirs:
				os.mkdir(d)
		if not os.path.exists("config.txt"):
			shutil.copyfile('sync/files/default_config.txt', 'config.txt')

	def refill_backtest():
		from Screener import Screener as screener
		try:
			historical_setups = pd.read_feather(
				r"C:\Stocks\local\study\historical_setups.feather")
		except:
			historical_setups = pd.DataFrame()
		if not os.path.exists("C:\Stocks\local\study\full_list_minus_annotated.feather"):
			shutil.copy(r"C:\Stocks\sync\files\full_scan.feather",
						r"C:\Stocks\local\study\full_list_minus_annotated.feather")
		while historical_setups.empty or (len(historical_setups[historical_setups["pre_annotation"] == ""]) < 2500):
			full_list_minus_annotation = pd.read_feather(
				r"C:\Stocks\local\study\full_list_minus_annotated.feather").sample(frac=1)
			screener.run(
				ticker=full_list_minus_annotation[:20]['ticker'].tolist(), fpath=0)
			full_list_minus_annotation = full_list_minus_annotation[20:].reset_index(
				drop=True)
			full_list_minus_annotation.to_feather(
				r"C:\Stocks\local\study\full_list_minus_annotated.feather")
			historical_setups = pd.read_feather(
				r"C:\Stocks\local\study\historical_setups.feather")

	def backup():
		date = datetime.date.today()
		src = r'C:/Stocks'
		dst = r'C:/Backups/' + str(date)
		shutil.copytree(src, dst)
		path = "C:/Backups/"
		dir_list = os.listdir(path)
		for b in dir_list:
			dt = datetime.datetime.strptime(b, '%Y-%m-%d')
			if (datetime.datetime.now() - dt).days > 30:
				shutil.rmtree((path + b))

	def add_setup(ticker, date, setup, val, req, ident=None):
		date = Data.format_date(date)
		add = pd.DataFrame({'ticker': [ticker], 'dt': [
			date], 'value': [val], 'required': [req]})
		if ident == None:
			ident = Data.get_config('Data identity') + '_'
		path = 'sync/database/' + ident + setup + '.feather'
		try:
			df = pd.read_feather(path)
		except FileNotFoundError:
			df = pd.DataFrame()
		df = pd.concat([df, add]).drop_duplicates(
			subset=['ticker', 'dt'], keep='last').reset_index(drop=True)
		df.to_feather(path)

	def consolidate_database():
		setups = Main.get_setups_list()
		for setup in setups:
			df = pd.DataFrame()
			# for ident in ['ben_','desktop_','laptop_', 'ben_laptop_']:
			for ident in ['desktop_', 'laptop_']:
				try:
					df1 = pd.read_feather(
						f"sync/database/{ident}{setup}.feather").dropna()
					df1['sindex'] = df1.index
					df1['source'] = ident
					df = pd.concat([df, df1]).reset_index(drop=True)
				except FileNotFoundError:
					pass
			df.to_feather(f"local/data/{setup}.feather")

	def get_setups_list():
		setups = []
		path = "sync/database/"
		dir_list = os.listdir(path)
		for p in dir_list:
			s = p.split('_')
			s = s[1] + '_' + s[2].split('.')[0]
			use = True
			for h in setups:
				if s == h:
					use = False
					break
			if use:
				setups.append(s)
		return setups


class Dataset:

	def np_worker(bar):
		df, type, bars = bar
		ds = df.load_np(type, bars)
		return df, ds

	def load_np(self, type, bars):
		self.np_type = type
		self.np_bars = bars
		arglist = [[df, type, bars] for df in self.dfs]
		lis = Dataset.try_pool(self, Dataset.np_worker, arglist)
	
		dfs = []
		returns = []
		for df, ds in lis:
			dfs.append(df)
			returns += ds
		self.dfs = dfs
		self.np = returns
		return returns
   # def __getattr__(self, name):
 #   lis = [Data.name]
 #   return []

	def try_pool(self, func, args):
		if current_process().name == 'MainProcess':
			return Main.pool(func, args)
		return [func(**arg) for arg in args]

	def train(self, st, percent_yes, epochs):
		df = pd.read_feather('C:/Stocks/local/data/' + st + '.feather')
		ones = len(df[df['value'] == 1])
		if ones < 150:
			print(f'{st} cannot be trained with only {ones} positives')
			return
		x = self.np
		y = self.np_y

		model = Sequential([Bidirectional(LSTM(64, input_shape=(x.shape[1], x.shape[2]), return_sequences=True,),), Dropout(
			0.2), Bidirectional(LSTM(32)), Dense(3, activation='softmax'),])
		model.compile(loss='sparse_categorical_crossentropy',
					  optimizer=Adam(learning_rate=1e-3), metrics=['accuracy'])
		model.fit(x, y, epochs=epochs, batch_size=64, validation_split=.2,)
		model.save('C:/Stocks/sync/models/model_' + st)
		tensorflow.keras.backend.clear_session()

	def init_worker(pack):
		bar, other = pack
		ticker = bar['ticker']
		dt = bar['dt']
		tf = bar['tf']
		bars, offset, value, pm, np_bars = other
		# needs to be fixed becaujse m
		return Data(ticker, tf, dt, bars, offset, value, pm, np_bars)

	def __init__(self, request=pd.DataFrame(), bars=0, offset=0, value=None, pm=True, np_bars=50):
		from Screener import Screener as screener
		if request.empty:
			tickers = screener.get('full')
			request = pd.DataFrame({'ticker': tickers})
			request['dt'] = None
			request['tf'] = 'd'
		other = (bars, offset, value, pm, np_bars)
		arglist = [[request.iloc[i], other] for i in range(len(request))]
		self.dfs = Dataset.try_pool(self,Dataset.init_worker, arglist)
		self.request = request
		self.len = len(self.dfs)
		self.value = value
		self.bars = bars
		self.offset = offset
		self.np_bars = np_bars


class Data:
	
	

	def update(self):
		ticker = self.ticker
		tf = self.tf
		exists = True
		try:
			df = feather.read_feather(Data.data_path(ticker,tf)).set_index('datetime',drop = True)######
			last_day = self.df.index[-1] 
		except: exists = False
		if tf == 'd':
			ytf = '1d'
			period = '25y'
		else:
			ytf = '1m'
			period = '5d'
		ydf = yf.download(tickers = ticker, period = period, group_by='ticker', interval = ytf, ignore_tz = True, progress=False, show_errors = False, threads = False, prepost = True) 
		ydf.drop(axis=1, labels="Adj Close",inplace = True)
		ydf.rename(columns={'Open':'open','High':'high','Low':'low','Close':'close','Volume':'volume'}, inplace = True)
		ydf.dropna(inplace = True)
		if Main.is_market_open() == 1: ydf.drop(ydf.tail(1).index,inplace=True)
		if not exists: df = ydf
		else:
			try: index = Data.findex(ydf, last_day) 
			except: return
			ydf = ydf[index + 1:]
			df = pd.concat([df, ydf])
		df.index.rename('datetime', inplace = True)
		if not df.empty: 
			if tf == '1min': pass
			elif tf == 'd': df.index = df.index.normalize() + pd.Timedelta(minutes = 570)
			df = df.reset_index()
			feather.write_feather(df,Data.data_path(ticker,tf))

	def __init__(self, ticker='QQQ', tf='d', dt=None, bars=0, offset=0, value=None, pm=True, np_bars=50):
		try:
			if len(tf) == 1:
				tf = '1' + tf
			dt = Main.format_date(dt)
			if 'd' in tf or 'w' in tf:
				base_tf = '1d'
			else:
				base_tf = '1min'
			try:
				df = feather.read_feather(Main.data_path(
					ticker, tf)).set_index('datetime', drop=True)
			except FileNotFoundError:
				raise TimeoutError
			if df.empty:
				raise TimeoutError
			if pm and Main.is_pre_market(dt):
				pm_bar = pd.read_feather(
					'sync/files/current_scan.feather').set_index('ticker').loc[ticker]
				pm_price = pm_bar['pm change'] + df.iat[-1, 3]
				df = pd.concat([df, pd.DataFrame({'datetime': [datetime.datetime.now()], 'open': [pm_price], 'high': [pm_price], 'low': [
					pm_price], 'close': [pm_price], 'volume': [pm_bar['pm volume']]}).set_index("datetime", drop=True)])
			if dt != None:
				try:
					df = df[:Data.findex(
							df, dt) + 1 + int(offset*(pd.Timedelta(tf) / pd.Timedelta(base_tf)))]
				except IndexError:
					raise TimeoutError
			if tf != '1min' or not pm:
				df = df.between_time('09:30', '15:59')
			if 'w' in tf:
				last_bar = df.tail(1)
				df = df[:-1]
			df = df.resample(tf, closed='left', label='left', origin=pd.Timestamp(
				'2008-01-07 09:30:00')).apply({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
			if 'w' in tf:
				df = pd.concat([df, last_bar])
			df = df.dropna()[-bars:]
		except TimeoutError:
			df = pd.DataFrame()
		self.df = df
		self.len = len(df)
		self.ticker = ticker
		self.tf = tf
		self.dt = dt
		self.value = value
		self.bars = bars
		self.offset = offset
		self.np_bars = np_bars

	def requirements(self):
		pass

	def load_np(self, type, bars):
		returns = []
		#bars = self.np_bars
		try:

			df = self.df
			if len(df) < 5:
				return returns
			if type == 'ml':
				partitions = 1
			else:
				partitions = bars//2
			x = df.to_numpy()
			x = np.flip(x, 0)

			d = np.zeros((x.shape[0]-1, x.shape[1]))
			for i in range(len(d)):  # add ohlc
				d[i] = x[i+1]/x[i, 3] - 1
			if partitions != 0:
				for i in list(range(bars, d.shape[0]+1, partitions)):
					if type == 'gpu':
						# x = x.reshape(1, 2, bars)
						# x = torch.tensor(list(x), requires_grad=True).cuda()
						# sequence2 = torch.tensor([1.0, 2.0, 2.5, 3.5], requires_grad=True).cuda()
						print('nice gpu code')
					else:
						x = d[i-bars:i]
						x = preprocessing.normalize(x, axis=0)
						if type != 'ml':
							x = x[:, 3]
							if type == 'dtw':
								x = np.column_stack(
									(x, numpy.arange(x.shape[0])))
						returns.append([x, i])
						#returns.append(x)
		except TimeoutError:
			pass
		self.np = returns
		return returns

	def load_plot(self, hidden=False):
		buffer = io.BytesIO()
		s = mpf.make_mpf_style(base_mpf_style='nightclouds', marketcolors=mpf.make_marketcolors(
			up='g', down='r', wick='inherit', edge='inherit', volume='inherit'))
		if hidden:
			title = ''
		else:
			title = f'{self.ticker}  {self.dt}  {self.tf}  {self.score}'
		if self.offset == 0:
			_, axlist = mpf.plot(self.df, type='candle', axisoff=True, volume=True, style=s, returnfig=True, title=title, figratio=(Main.get_config(
				'Study chart_aspect_ratio'), 1), figscale=Main.get_config('Study chart_size'), panel_ratios=(5, 1),  tight_layout=True, vlines=dict(vlines=[self.dt], alpha=.25))
		else:
			_, axlist = mpf.plot(df, type='candle', volume=True, axisoff=True, style=s, returnfig=True, title=title, figratio=(Main.get_config(
				'Study chart_aspect_ratio'), 1), figscale=Main.get_config('Study chart_size'), panel_ratios=(5, 1),  tight_layout=True)
		ax = axlist[0]
		ax.set_yscale('log')
		ax.yaxis.set_minor_formatter(mticker.ScalarFormatter())
		plt.savefig(buffer, format='png')
		self.plot = buffer.getvalue()
		return self.plot

	def load_score(self, st, model=None):
		if model == None:
			model = Main.load_model(st)
		returns = []
		for df, index in self.np:
			score = model.predict(df)
			returns.append([self.ticker, self.df.index[index], st, score])
		self.score = returns
		return returns

	def findex(self, dt):
		dt = Main.format_date(dt)
		if not isinstance(self, pd.DataFrame):
			df = self.df
		else:
			df = self
		i = int(len(df)/2)
		k = int(i/2)
		while k != 0:
			date = df.index[i].to_pydatetime()
			if date > dt:
				i -= k
			elif date < dt:
				i += k
			k = int(k/2)
		while df.index[i].to_pydatetime() < dt:
			i += 1
		while df.index[i].to_pydatetime() > dt:
			i -= 1
		return i





if __name__ == '__main__':
	Main.run()
