import discord
import yfinance as yf
import pandas as pd
from pandas_datareader import data as pdr
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from io import BytesIO
from func import *

import os
from dotenv import load_dotenv

load_dotenv()

intents = discord.Intents.default()
client = discord.Client(intents=intents)


def ticker_(stock):
  return yf.Ticker(stock)


# stock=input("Enter stock: ")
ticker = ticker_("RELIANCE.NS")

Monthly_history = ticker.history(period='1d', interval='1m')
latest_price = Monthly_history['Close'].iloc[-1]
Monthly_history.to_csv("temp.csv")

df = pd.read_csv("temp.csv")
df["Datetime"] = pd.to_datetime(df["Datetime"])
df["hour"] = df["Datetime"].dt.hour
df["minute"] = df["Datetime"].dt.minute


@client.event
async def on_ready():
  print('We have logged in as {0.user}'.format(client))


@client.event
async def on_message(message):
  if message.content.startswith('help'):
    await message.channel.send(
        "Type 'price' to get the last 1d graph of the stock: RELIANCE.NS")
  elif message.author == client.user:
    return
  elif message.content.startswith('hello'):
    await message.channel.send('Hello! Have a nice DAY!!')

  elif message.content.startswith('price'):
    await message.channel.send('Wait a sec')
    buffer = show_plot()

    # Send the plot as an image
    await message.channel.send(file=discord.File(buffer, 'plot.png'))
  else:
    await message.channel.send('Try typing "help"!!')


# client.run(os.getenv('TOKEN'))
load_dotenv(dotenv_path="1.env")

# print(os.getenv('TOKEN'))  # Add this line to check the value of the 'TOKEN' environment variable
my_secret = os.environ['TOKEN']

client.run(my_secret)
