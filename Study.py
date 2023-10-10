import pandas as pd
import mplfinance as mpf
import PySimpleGUI as sg
from Data import Data as data
import matplotlib.ticker as mticker
from matplotlib import pyplot as plt
from multiprocessing.pool import Pool
import os, pathlib, shutil, math, PIL, io

class Study:

    def run(self,current):
        self.current = current
        if not self.current:
            self.sub_st_list = {}
            setups_df = pd.read_feather(r"C:\Stocks\local\study\historical_setups.feather")
            for st in data.get_setups_list():
                df = setups_df[setups_df['st'] == st]
                sub_st_list = [*set(df['sub_st'].to_list())]
                self.sub_st_list.update({st:[s for s in sub_st_list if s != st]})
        with Pool(int(data.get_config('Data cpu_cores'))) as self.pool:
            self.init = True
            self.filter(self)
            while True:
                self.event, self.values = self.window.read()
                if self.event in ['Yes','No']:
                    if self.event == 'Yes': v = 1
                    else: v = 0
                    bar = self.setups_data.iloc[math.floor(self.i)]
                    ticker = bar['ticker']
                    dt = bar['dt']
                    if data.is_pre_market(dt): dt = dt.replace(hour=9, minute=30)
                    st = bar['st']
                    data.add_setup(ticker,dt,st,v,1)
                    self.event = 'Next'
                if self.event == 'Next' and (self.i < len(self.setups_data) - 1 or (self.i < len(self.setups_data) - .5 and not self.current)): 
                    self.previ = self.i
                    if self.current: self.i += 1
                    else: self.i += .5
                    self.update(self)
                    self.window.refresh()
                    self.preload(self)
                elif self.event == 'Prev' and self.i > 0: 
                    self.previ = self.i
                    if self.current: self.i -= 1
                    else: self.i -= .5
                    self.update(self)
                elif self.event == 'Load':
                    self.previ = self.i 
                    self.update(self)
                    self.filter(self)
                if self.event == sg.WIN_CLOSED:
                    self.previ = self.i
                    self.update(self)
                    self.window.close()

    def preload(self):
        preload_amount = 10
        if self.i == 0:
            if self.current: index_list = [float(i) for i in range(preload_amount)]
            else: index_list = [float(i/2) for i in range(preload_amount*2)]
        else: index_list = [preload_amount-1 + self.i]
        arglist = [[self.setups_data,i,self.current] for i in index_list if i < len(self.setups_data)]
        self.pool.map_async(self.plot,arglist)
        
    def filter(self):
        try:
            if self.current: 
                try: df = pd.read_feather(r"C:\Stocks\local\study\current_setups.feather").sort_values(by=['z'], ascending=False)
                except: raise Exception('No current setups found')
            else:
                df = pd.read_feather(r"C:\Stocks\local\study\historical_setups.feather")
                sort_by = None
                if not self.init:
                    input_filter = self.values['-input_filter-']
                    reqs = input_filter.split('&')
                    if input_filter != "":
                        for req in reqs:
                            if '^' in req:
                                sort_by = req.split('^')[1]
                                if sort_by not in df.columns: raise TimeoutError
                            else:
                                r = req.split('=')
                                trait = r[0]
                                if trait not in df.columns or len(r) == 1: raise TimeoutError
                                val = r[1]
                                if 'annotation' in trait: df = df[df[trait].str.contains(val)]
                                else: df = df[df[trait] == val]
                if sort_by != None: df = df.sort_values(by = [sort_by], ascending = False)
                else: df = df.sample(frac = 1)
            if df.empty: raise TimeoutError
            self.setups_data = df
            self.i = 0.0
            self.previ = None
            while os.path.exists("C:/Stocks/local/study/charts"):
                try:shutil.rmtree("C:/Stocks/local/study/charts")
                except: pass
            os.mkdir("C:/Stocks/local/study/charts")
            self.preload(self)
            self.update(self)
        except TimeoutError: sg.Popup('no setups found')

    def plot(bar):
        setups_data = bar[0]
        i = bar[1]
        current = bar[2]
        if int(i) != i: revealed = True
        else: revealed = False
        bar = setups_data.iloc[math.floor(i)]
        dt = data.format_date(bar['dt'])
        ticker = bar['ticker']
        st = bar['st']
        z = bar['z']
        tf = st.split('_')[0]
        tf_list = []
        if 'w' in tf or 'd' in tf or 'h' in tf:
            intraday = False
            req_tf = ['1min','h','d','w']
            for t in req_tf:
                if t in tf: tf_list.append(tf)
                else: tf_list.append(t)
        else:
            intraday == True
            if tf == '1min': tf_list = ['d','h','5min','1min']
            else: tf_list = ['d','h',tf,'1min']
        plt.rcParams.update({'font.size': 30})
        ii = len(tf_list)
        first_minute_high = 1
        first_minute_low = 1
        first_minute_close = 1
        first_minute_volume = 0
        s = mpf.make_mpf_style(base_mpf_style= 'nightclouds',marketcolors=mpf.make_marketcolors(up='g',down='r',wick ='inherit',edge='inherit',volume='inherit'))
        for tf in tf_list:
            p = pathlib.Path("C:/Stocks/local/study/charts") / f'{ii}_{i}.png'
            try:
                chart_size = 100
                if 'min' in tf: chart_offset = chart_size - 1
                else: chart_offset = 20
                if not revealed: chart_offset = 0
                df = data.get(ticker,tf,dt,chart_size,chart_offset)
                if df.empty: raise TimeoutError
                if not revealed and not intraday:
                    if tf == '1min':
                        open = df.iat[-1,0]
                        first_minute_high = df.iat[-1,1]/open
                        first_minute_low = df.iat[-1,2]/open
                        first_minute_close = df.iat[-1,3]/open
                        first_minute_volume = df.iat[-1,4]
                    else:
                        open = df.iat[-1,0]
                        df.iat[-1,1] = open * first_minute_high
                        df.iat[-1,2] = open * first_minute_low
                        df.iat[-1,3] = open * first_minute_close
                        df.iat[-1,4] = first_minute_volume
                if (current or revealed) and ii == 1: title = f'{ticker} {dt} {st} {z} {tf}' 
                else: title = str(tf)
                #if revealed: _, axlist = mpf.plot(df, type='candle', axisoff=True,volume=True, style=s, returnfig = True, title = title, figratio = (data.get_config('Study chart_aspect_ratio'),1),figscale=data.get_config('Study chart_size'), panel_ratios = (5,1), mav=(10,20), tight_layout = True,vlines=dict(vlines=[dt], alpha = .25))
                #else: _, axlist =  mpf.plot(df, type='candle', volume=True,axisoff=True,style=s, returnfig = True, title = title, figratio = (data.get_config('Study chart_aspect_ratio'),1),figscale=data.get_config('Study chart_size'), panel_ratios = (5,1), mav=(10,20), tight_layout = True)
                if revealed: _, axlist = mpf.plot(df, type='candle', axisoff=True,volume=True, style=s, returnfig = True, title = title, figratio = (data.get_config('Study chart_aspect_ratio'),1),figscale=data.get_config('Study chart_size'), panel_ratios = (5,1),  tight_layout = True,vlines=dict(vlines=[dt], alpha = .25))
                else: _, axlist =  mpf.plot(df, type='candle', volume=True,axisoff=True,style=s, returnfig = True, title = title, figratio = (data.get_config('Study chart_aspect_ratio'),1),figscale=data.get_config('Study chart_size'), panel_ratios = (5,1),  tight_layout = True)
                
                ax = axlist[0]
                ax.set_yscale('log')
                ax.yaxis.set_minor_formatter(mticker.ScalarFormatter())
                plt.savefig(p, bbox_inches='tight',dpi = data.get_config('Study chart_dpi'))
            except: shutil.copy(r"C:\Stocks\sync\files\blank.png",p)
            ii -= 1

    def update(self):
        if self.init:
            sg.theme('Black')
            layout = [[sg.Image(key = '-chart1-'),sg.Image(key = '-chart2-')],
            [sg.Image(key = '-chart3-'),sg.Image(key = '-chart4-')],
            [(sg.Text(key = '-counter-'))]]
            if self.current: layout += [[sg.Button('Prev'), sg.Button('Next')]]
            #if self.current: layout += [[sg.Button('Prev'), sg.Button('Next'), sg.Button('Yes'),sg.Button('No')]]
            else: 
                df = pd.read_feather(r"C:\Stocks\local\study\historical_setups.feather")
                self.annotated = len(df[df['pre_annotation'] != ''])
                layout += [[sg.Multiline(size=(150, 5), key='-annotation-')],[sg.Combo([],key = '-sub_st-', size = (20,10))],[sg.Button('Prev'), sg.Button('Next'),sg.Button('Yes'),sg.Button('No'),sg.Button('Load'),sg.InputText(key = '-input_filter-'),sg.Text(key='annotated')]]
            self.window = sg.Window('Study', layout,margins = (10,10),scaling = data.get_config('Study ui_scale'),finalize = True)
            self.init = False
        for i in range(1,5):
            while True:
                try: 
                    image = PIL.Image.open(f'C:\Stocks\local\study\charts\{i}_{self.i}.png')
                    bio = io.BytesIO()
                    image.save(bio,format="PNG")
                    self.window[f'-chart{i}-'].update(data = bio.getvalue())
                    break
                except (PIL.UnidentifiedImageError, FileNotFoundError, OSError): pass
        self.window['-counter-'].update(str(f"{math.floor(self.i + 1)} of {len(self.setups_data)}"))
        if not self.current:
            if self.previ != None:
                df = pd.read_feather(r"C:\Stocks\local\study\historical_setups.feather")
                annotation = self.values["-annotation-"]
                sub_st = self.values['-sub_st-']
                st = self.setups_data.iloc[math.floor(self.i)]['st']
                if sub_st != st and sub_st not in self.sub_st_list[st]: self.sub_st_list[st].append(sub_st)
                if int(self.previ) == self.previ: col = 'pre_annotation'
                else: col = 'post_annotation'
                index = self.setups_data.index[math.floor(self.previ)]
                self.setups_data.at[index, col] = annotation
                df.at[index, col] = annotation
                self.setups_data.at[index,'sub_st'] = sub_st
                df.at[index,'sub_st'] = sub_st
                df.to_feather(r"C:\Stocks\local\study\historical_setups.feather")
            if int(self.i) == self.i: 
                self.annotated += 1
                self.window['annotated'].update(str(self.annotated))
                col = 'pre_annotation'
            else: col = 'post_annotation'
            bar = self.setups_data.iloc[math.floor(self.i)]
            self.window["-annotation-"].update(bar[col])
            ss = list(self.sub_st_list[bar['st']])
            self.window['-sub_st-'].update(values = ss, value = bar['sub_st'])
        self.window.maximize()

if __name__ == "__main__":
    Study.run(Study,False)