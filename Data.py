import numpy as np
import websocket, datetime, os, pyarrow, shutil,statistics, warnings, math, time, pytz, tensorflow, random
from pyarrow import feather
import asyncio
from multiprocessing import Pool
import pandas as pd

class data:
	
	def __init__(self,ticker = 'qqq',tf = 'd',dt = None,bars = 0,offset = 0,value = None):
		self.ticker = ticker
		self.tf = tf
		self.dt = dt
		self.value = value
		self.bars = bars
		self.offset = offset
		self.scores = []
		try:



			if len(tf) == 1: tf = '1' + tf # For example, if tf='d', tf = '1d' 
			dt = main.format_date(dt) #
			if 'd' in tf or 'w' in tf: base_tf = '1d'
			else: base_tf = '1min'
			try: df = feather.read_feather(main.data_path(ticker,tf)).set_index('datetime',drop = true)
			except FileNotFoundError: df = pd.dataframe()


			#if (df.empty or (dt != none and (dt < df.index[0] or dt > df.index[-1]))) and not (base_tf == '1d' and main.is_pre_market(dt)): 
		#		try: 
		#			add = tvdatafeed(username="cs.benliu@gmail.com",password="tltshort!1").get_hist(ticker,pd.read_feather('c:/stocks/sync/files/full_scan.feather').set_index('ticker').loc[ticker]['exchange'], interval=base_tf, n_bars=100000, extended_session = main.is_pre_market(dt))
		#			add.iloc[0]
			#	except: pass
			#	else:
			#		add.drop('symbol', axis = 1, inplace = true)
			#		add.index = add.index + pd.timedelta(hours=(13-(time.timezone/3600)))
			#		if df.empty or add.index[0] > df.index[-1]: df = add
				#	else: df = pd.concat([df,add[main.findex(add,df.index[-1]) + 1:]])
			if df.empty: raise TimeoutError
			if dt != None and not main.is_pre_market(dt):
				try: df = df[:data.findex(df,dt) + 1 + int(offset*(pd.timedelta(tf) / pd.timedelta(base_tf)))]
				except indexerror: raise timeouterror
			if 'min' not in tf and base_tf == '1min': df = df.between_time('09:30', '15:59')##########
			if 'w' in tf and not main.is_pre_market(dt):
				last_bar = df.tail(1)
				df = df[:-1]
			df = df.resample(tf,closed = 'left',label = 'left',origin = pd.timestamp('2008-01-07 09:30:00')).apply({'open':'first','high':'max','low':'min','close':'last','volume':'sum'})
			if 'w' in tf and not main.is_pre_market(dt): df = pd.concat([df,last_bar])
			if base_tf == '1d' and main.is_pre_market(dt): 
				pm_bar = pd.read_feather('c:/stocks/sync/files/current_scan.feather').set_index('ticker').loc[ticker]
				pm_price = pm_bar['pm change'] + df.iat[-1,3]
				df = pd.concat([df,pd.dataframe({'datetime': [dt], 'open': [pm_price],'high': [pm_price], 'low': [pm_price], 'close': [pm_price], 'volume': [pm_bar['pm volume']]}).set_index("datetime",drop = true)])
			df = df.dropna()[-bars:]
		except TimeoutError:
			df = pd.dataframe()
			
		self.df = df
		self.len = len(df)
				 


async def worker(ticker,tf):  ###worker for the main fucntion. this will be multiprcoessed by main.
	with open(f'local/data/{tf}/{ticker}.npy', 'rb') as file:
		return np.load(file)

async def main(tickers,tf): #this runs the fast_get pool using async. Dont really understand this it is chatgpt code
	loop = asyncio.get_event_loop()
	with Pool() as pool:
		tasks = [loop.run_in_executor(pool, worker, ticker + '/' + tf) for ticker in tickers]
		results = await asyncio.gather(*tasks)
	return results




def fast_get(tickers,tf = 'd'):
	result = asyncio.run(worker(tickers,tf))
	return result


if __name__ == '__main__':
	df1 = pd.read_feather("C:/Stocks/sync/files/full_scan.feather")
	ticker_list = df1['ticker'].tolist()
	start = datetime.datetime.now()
	fast_get(ticker_list)
	print(datetime.datetime.now() - start)






























