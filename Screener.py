from re import S
import pandas as pd
import numpy as np
import mplfinance as mpf
from Data import Data as data
from multiprocessing import Pool
from Study import Study as study
from discordwebhook import Discord
import selenium.webdriver as webdriver
from selenium.webdriver.common.by import By 
from selenium.webdriver.firefox.options import Options 
from selenium.webdriver.firefox.service import Service
import pathlib, time, selenium, datetime, os, math, tensorflow

class Screener:

	def run(dt = None, ticker = None, tf = 'd',browser = None, fpath = None):
		with Pool(int(data.get_config('Data cpu_cores'))) as pool:
			dt = data.format_date(dt)
			path = 0
			if ticker == None:
				if dt == None: 
					path = 0
					ticker_list = Screener.get('full')
				else:
					if 'd' in tf or 'w' in tf: 
						ticker_list, browser = Screener.get('current',True,browser)
						path = 1
					else: 
						ticker_list, browser = Screener.get('intraday',True,browser)
						path = 2
			else:
				path = 1
				if not isinstance(ticker,list): ticker = [ticker]
				ticker_list = ticker
			if fpath != None: path = fpath
			if path == 1: 
				try: os.remove(r"C:\Stocks\local\study\current_setups.feather")
				except FileNotFoundError: pass
			df = pd.DataFrame()
			df['ticker'] = ticker_list
			df['dt'] = dt
			df['tf'] = tf
			num_dfs = int(data.get_config('Data cpu_cores'))
			values = []
			s = math.ceil(len(df) / num_dfs)
			for i in range(num_dfs):
				d = df[int(s*i):int(s*(i+1))].reset_index(drop = True)
				if not d.empty: values.append(pool.apply_async(data.create_arrays,args = ([d]))) 
			tensorflow.keras.backend.clear_session()
			model_list = [[st,data.load_model(st)] for st in data.get_config('Screener active_setup_list').split(',') if tf in st]
			x = np.concatenate([bar.get()[0] for bar in values])
			info = np.concatenate([bar.get()[2] for bar in values])
			dfs = {}
			for bar in values: dfs.update(bar.get()[3])
			for st, model in model_list:
				setups = data.score(x,info,dfs,st,model)
				for ticker,dt,score,df in setups:
					Screener.log(ticker,score,dt,tf,path,st,df)

	def log(ticker,z, dt, tf,  path, setup_type,df):
		dt = data.format_date(dt)
		if path == 3: print(f'{ticker} {dt} {z} {setup_type}')
		elif path == 2:
			mpf.plot(df[-100:], type='candle', mav=(10, 20), volume=True, title=f'{ticker}, {setup_type}, {z}, {tf}', style=mpf.make_mpf_style(marketcolors=mpf.make_marketcolors(up='g',down='r')), savefig = pathlib.Path("C:/Stocks/local/screener")/ 'intraday.png')
			Discord(url="https://discord.com/api/webhooks/1071667193709858847/qwHcqShmotkEPkml8BSMTTnSp38xL1-bw9ESFRhBe5jPB9o5wcE9oikfAbt-EKEt7d3c").post(file={"intraday": open('local/screener/intraday.png', "rb")})
		elif path == 1:
			d = r"C:\Stocks\local\study\current_setups.feather"
			try: setups = pd.read_feather(d)
			except: setups = pd.DataFrame()
			setups = pd.concat([setups,pd.DataFrame({'ticker': [ticker],'dt':[dt],'st': [setup_type],'z':[z]})]).reset_index(drop = True)
			setups.to_feather(d)
		elif path == 0:
			d = r"C:\Stocks\local\study\historical_setups.feather"
			try: setups = pd.read_feather(d)
			except: setups = pd.DataFrame()
			setups = pd.concat([setups,pd.DataFrame({'ticker':[ticker], 'dt': [dt],'st': [setup_type], 'z': [z], 'sub_st':[setup_type], 'pre_annotation': [""], 'post_annotation': [""] })]) .reset_index(drop = True)
			setups.to_feather(d)

	def get(type = 'full', refresh = False, browser = None):

		def start_firefox():
			options = webdriver.FirefoxOptions()
			options.binary_location = r"C:\Program Files\Mozilla Firefox\firefox.exe"
			options.headless = True
			service = Service(executable_path=os.path.join(os.getcwd(), 'Drivers', 'geckodriver.exe'))
			FireFoxProfile = webdriver.FirefoxProfile()
			FireFoxProfile.set_preference("General.useragent.override", 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:87.0) Gecko/20100101 Firefox/87.0')
			browser = webdriver.Firefox(options=options, service=service)
			browser.implicitly_wait(7)
			browser.set_window_size(2560, 1440)
			browser.get("https://www.tradingview.com/screener/")
			time.sleep(1.5)
			browser.find_element(By.XPATH, '//button[@aria-label="Open user menu"]').click()
			time.sleep(1)
			browser.find_element(By.XPATH, '//button[@data-name="header-user-menu-sign-in"]').click()
			time.sleep(1)
			browser.find_element(By.XPATH, '//button[@class="emailButton-nKAw8Hvt light-button-bYDQcOkp with-start-icon-bYDQcOkp variant-secondary-bYDQcOkp color-gray-bYDQcOkp size-medium-bYDQcOkp typography-regular16px-bYDQcOkp"]').click()
			browser.find_element(By.XPATH, '//input[@name="id_username"]').send_keys("cs.benliu@gmail.com")
			time.sleep(0.5)
			browser.find_element(By.XPATH, '//input[@name="id_password"]').send_keys("tltShort!1")
			time.sleep(0.5)
			browser.find_element(By.XPATH, '//button[@class="submitButton-LQwxK8Bm button-D4RPB3ZC size-large-D4RPB3ZC color-brand-D4RPB3ZC variant-primary-D4RPB3ZC stretch-D4RPB3ZC apply-overflow-tooltip apply-overflow-tooltip--check-children-recursively apply-overflow-tooltip--allow-text"]').click()
			time.sleep(3)
			browser.refresh()
			time.sleep(5)
			browser.find_element(By.XPATH, '//div[@data-name="screener-field-sets"]').click()
			time.sleep(0.1)
			browser.find_element(By.XPATH, '//div[@title="Python Screener"]').click()
			filter_tab = browser.find_element(By.XPATH, '//div[@class="tv-screener-sticky-header-wrapper__fields-button-wrap"]')
			try: filter_tab.click()
			except: pass
			time.sleep(0.5)
			browser.find_element(By.XPATH, '//div[@class="tv-screener__standalone-title-wrap"]').click()
			time.sleep(0.5) 
			browser.find_element(By.XPATH, '//div[@data-name="screener-filter-sets"]').click()
			time.sleep(0.25)
			browser.find_element(By.XPATH, '//span[@class="js-filter-set-name"]').click()
			time.sleep(0.25)
			browser.find_element(By.XPATH, '//div[@data-field="relative_volume_intraday.5"]').click()
			return browser

		def get_full(refresh):
			df1 = pd.read_feather("C:/Stocks/sync/files/full_scan.feather")
			if not refresh: return df1['ticker'].tolist()
			df2 = pd.read_feather("C:/Stocks/sync/files/current_scan.feather")
			df3 = pd.concat([df1,df2]).drop_duplicates(subset = ['ticker'])		
			not_in_current = (pd.concat([df3,df2]).drop_duplicates(subset = ['ticker'],keep = False))['ticker'].tolist()
			removelist = []
			for ticker in not_in_current:
				if pd.isna(ticker) or not os.path.exists('C:/Stocks/local/data/1min/' + ticker + '.feather'):
					removelist.append(ticker)
			df3 = df3.set_index('ticker',drop = True)
			df3.drop(removelist, inplace = True)
			df3 = df3.reset_index()
			df3.to_feather("C:/Stocks/sync/files/full_scan.feather")
			return df3['ticker'].tolist()

		def get_current(refresh,browser = None):
			if not refresh:
				try: return pd.read_feather("C:/Stocks/sync/files/current_scan.feather")['ticker'].tolist(), browser
				except FileNotFoundError: pass
			try:
				if browser == None: browser = start_firefox()
				time.sleep(0.5) 
				browser.find_element(By.XPATH, '//div[@data-name="screener-filter-sets"]').click()
				time.sleep(0.25)
				browser.find_element(By.XPATH, '//span[@class="js-filter-set-name"]').click()
				time.sleep(0.25)
				browser.find_element(By.XPATH, '//div[@data-field="relative_volume_intraday.5"]').click()
				browser.find_element(By.XPATH, '//div[@data-name="screener-export-data"]').click()
			except Exception as e: 
				print(e)
				print('manual csv fetch required')
			found = False
			today = str(datetime.date.today())
			while True:
				path = r'C:/Downloads/'
				dir_list = os.listdir(path)
				for direct in dir_list:
					if today in direct:
						downloaded_file = path + direct
						found = True
						time.sleep(1)
						break
				if found:
					break
			df = pd.read_csv(downloaded_file)
			os.remove(downloaded_file)
			for i in range(len(df)-1,-1,-1):
				bar = df.loc[i]
				ticker = bar['Ticker']
				if  isinstance(ticker,str) and '.' not in ticker and '/' not in ticker and not ticker == 'nan':
					if str(bar['Exchange']) == "NYSE ARCA": df.iloc[i]['Exchange'] = "AMEX"
				else: df = df.drop(i)
			df = df.drop('Description', axis = 1)
			df = df.fillna(0)
			df = df.rename(columns={'Ticker':'ticker','Exchange':'exchange','Pre-market Change':'pm change','Pre-market Volume':'pm volume','Relative Volume at Time':'rvol'})
			df = df.reset_index(drop = True)
			df.to_feather(r"C:\Stocks\sync\files\current_scan.feather")
			return df['ticker'].tolist() , browser

		def get_intraday(browser = None):
			while True:
				try:
					get_current(True,browser)
					df = pd.read_feather("C:/Stocks/sync/files/current_scan.feather")
					break
				except (selenium.common.exceptions.NoSuchElementException, AttributeError):
					try: browser.find_element(By.XPATH, '//button[@class="close-button-FuMQAaGA closeButton-zCsHEeYj defaultClose-zCsHEeYj"]').click()
					except (selenium.common.exceptions.NoSuchElementException, AttributeError): pass
			df = df.sort_values('rvol', ascending = False)
			df = df[:100].reset_index(drop = True)
			return df['ticker'].tolist(), browser

		if type == 'full': return get_full(refresh)
		elif type == 'current': return get_current(refresh,browser)
		elif type == 'intraday': return get_intraday(browser)

if __name__ == '__main__':
	Screener.run('current')
	study.run(study,True)
