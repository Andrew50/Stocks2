from torch.utils.data import DataLoader, TensorDataset
import asyncio, torch, datetime
from Screener import Screener as screener
from soft_dtw_cuda import SoftDTW
from tqdm import tqdm
from Test import Main, Data

async def process_batch(batch, sdtw):
    x, y_list = batch
    y = y_list[0]
    returns = []
    x = torch.from_numpy(x).float().unsqueeze(0).cuda()
    y = torch.from_numpy(y.numpy()).float().unsqueeze(0).cuda()
    loss = sdtw(x, y)
    return[loss.mean().item()]

async def main(dateloader):
    scores = []
    sdtw = SoftDTW(use_cuda=True, gamma=0.1)
    pbar = tqdm(total=len(dataloader.dataset))
    for batch in dataloader:
        x_batch, y = batch
        result = await asyncio.gather(process_batch((x, y), sdtw))
        scores.extend(result[0])
        pbar.update(x_batch.shape[0])
    pbar.close()
    return scores
    
def fetch(ticker,bars=10,dt = None):
    tf = 'd'
    if dt != None:
        df = Data(ticker,tf,dt,bars = bars+1)
    else:
        df = Data(ticker,tf)
    df.np(bars,True)
    return df

if __name__ == '__main__':
    ticker_list = screener.get('full')
    dfs = Main.pool(fetch, ticker_list)
    x_list = []
    for df in dfs:
        ticker = df.ticker
        for x, index in df.np:
            x_list.append(x) 
    ticker = 'JBL'  # input('input ticker: ')
    dt = '2023-10-03'  # input('input date: ')
    bars = 10  # int(input('input bars: '))
    start = datetime.datetime.now()
    batch_size = 100
    num_workers = 8
    y = fetch(ticker, bars, dt).np[0][0]
    y = [y for _ in range(len(x_list))]



    dataset = TensorDataset(torch.tensor(x_list), torch.tensor(y))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    scores = asyncio.run(main(dataloader))
    scores.sort(key=lambda x: x)
    print(f'completed in {datetime.datetime.now() - start}')
    for score in scores[:20]:
        print(score)
