import numpy as np
import websocket, datetime, os, pyarrow, shutil,statistics, warnings, math, time, pytz, tensorflow, random
from pyarrow import feather
import asyncio
from multiprocessing import Pool
from Screener import Screener as screener

class Data:
	
	def __init__(self,ticker = 'QQQ',tf = 'd',dt = None,bars = 0,offset = 0,value = None):
		self.ticker = ticker
		self.tf = tf
		self.dt = dt
		self.value = value
		self.bars = bars
		self.offset = offset
		self.scores = []
		try:



			if len(tf) == 1: tf = '1' + tf # (Eg: )
			dt = Main.format_date(dt)
			if 'd' in tf or 'w' in tf: base_tf = '1d'
			else: base_tf = '1min'
			try: df = feather.read_feather(Main.data_path(ticker,tf)).set_index('datetime',drop = True)
			except FileNotFoundError: df = pd.DataFrame()


			#if (df.empty or (dt != None and (dt < df.index[0] or dt > df.index[-1]))) and not (base_tf == '1d' and Main.is_pre_market(dt)): 
		#		try: 
	#				add = TvDatafeed(username="cs.benliu@gmail.com",password="tltShort!1").get_hist(ticker,pd.read_feather('C:/Stocks/sync/files/full_scan.feather').set_index('ticker').loc[ticker]['exchange'], interval=base_tf, n_bars=100000, extended_session = Main.is_pre_market(dt))
		#			add.iloc[0]
			#	except: pass
			#	else:
			#		add.drop('symbol', axis = 1, inplace = True)
			#		add.index = add.index + pd.Timedelta(hours=(13-(time.timezone/3600)))
			#		if df.empty or add.index[0] > df.index[-1]: df = add
				#	else: df = pd.concat([df,add[Main.findex(add,df.index[-1]) + 1:]])
			if df.empty: raise TimeoutError
			if dt != None and not Main.is_pre_market(dt):
				try: df = df[:Data.findex(df,dt) + 1 + int(offset*(pd.Timedelta(tf) / pd.Timedelta(base_tf)))]
				except IndexError: raise TimeoutError
			if 'min' not in tf and base_tf == '1min': df = df.between_time('09:30', '15:59')##########
			if 'w' in tf and not Main.is_pre_market(dt):
				last_bar = df.tail(1)
				df = df[:-1]
			df = df.resample(tf,closed = 'left',label = 'left',origin = pd.Timestamp('2008-01-07 09:30:00')).apply({'open':'first','high':'max','low':'min','close':'last','volume':'sum'})
			if 'w' in tf and not Main.is_pre_market(dt): df = pd.concat([df,last_bar])
			if base_tf == '1d' and Main.is_pre_market(dt): 
				pm_bar = pd.read_feather('C:/Stocks/sync/files/current_scan.feather').set_index('ticker').loc[ticker]
				pm_price = pm_bar['pm change'] + df.iat[-1,3]
				df = pd.concat([df,pd.DataFrame({'datetime': [dt], 'open': [pm_price],'high': [pm_price], 'low': [pm_price], 'close': [pm_price], 'volume': [pm_bar['pm volume']]}).set_index("datetime",drop = True)])
			df = df.dropna()[-bars:]
		except TimeoutError:
			df = pd.DataFrame()
			
		self.df = df
		self.len = len(df)
				 


async def worker(bar):
    ticker, tf = bar
    with open(f'local/data/{tf}/{ticker}.npy', 'rb') as file:
        return np.load(file)

async def main(tickers,tf):
    loop = asyncio.get_event_loop()
    with Pool() as pool:
        tasks = [loop.run_in_executor(pool, worker, [ticker,tf]) for ticker in tickers]
        results = await asyncio.gather(*tasks)
    return results




def fast_get(tickers,tf = 'd'):
	result = asyncio.run(worker(tickers,tf))
	return result


if __name__ == '__main__':
	ticker_list = screener.get('full')
	start = datetime.datetime.now()
	fast_get(ticker_list)
	print(datetime.datetime.now() - start)






























