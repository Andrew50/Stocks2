import discord
import discordwebhook
from Data import Data as data
from Match import Match as match 
from Test import Data
class discordManager(): 
	def initialize():
		intents = discord.Intents.default()
		intents.message_content = True
		client = discord_bot(intents=intents)
		client.run('MTE2MDAyNzYzMzY1Mjg3NTMyNQ.GDB8fq.ELueCj6j1lGnYE0iLGIHUD51hzWm_JdyqaNrK0')
class discord_bot(discord.Client): 
	
	def start_query(message, ticker, dt, numBars):
		returnedScores = match.initiate(ticker, dt, numBars)
		returnMessage = ""
		for ticker,index,score in returnedScores:
			returnMessage = returnMessage + (f'{ticker} {Data(ticker).df.index[index]} \n')
			message.channel.send(returnMessage)
	async def on_ready(self):
		pass
	async def on_message(self, message):
		message_content = message.content
		message_author = message.author
		if message.content.startswith('query?'):
			await message.channel.send('Query Initiated')
			text = message_content.split('? ')[1].split(' ')
			try:
				ticker = text[0]
				dt = text[1]
				numBars = int(text[2])
				discord_bot.start_query(message, ticker, dt, numBars)
			except TimeoutError:
				await message.channel.send('Incorrect format used.')
				
if __name__ == '__main__':
	discordManager.initialize()

				

