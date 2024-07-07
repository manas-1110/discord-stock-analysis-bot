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

df["sma_5m"]=df["Close"]

def stat(df):
    for i in range(len(df)-1,3,-1):
      df.iloc[i,10]=(df.iloc[i,4]+df.iloc[i-1,4]+df.iloc[i-2,4]+df.iloc[i-3,4]+df.iloc[i-4,4])/5


stat(df)

df["sd_5m"]=df["sma_5m"]-df["Close"]

df["bub"]=df["Close"]+2*df["sd_5m"]

df["blb"]=df["Close"]-2*df["sd_5m"]


def show_plot():
    columns = ["Datetime", "Close"]
    plt.figure(figsize=(17, 4))

    df = pd.read_csv("temp.csv", usecols=columns)
    plt.plot(df.Datetime, df.Close)
    plt.plot(df["Datetime"],df["sma_5m"])
    plt.plot(df["Datetime"],df["bub"])
    plt.plot(df["Datetime"],df["blb"])

    # Save the plot to a BytesIO object
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()

    return buffer

def lineplot():
  hr=int(input("enter hour: "))
  if(hr<9 or hr>16):
    print("invalid time!")
  else:
    x=df.loc[df["hour"]==hr]["minute"]
    y=df.loc[df["hour"]==hr]["Close"]
    a,b=np.polyfit(x.to_numpy(),y.to_numpy(),1)
    plt.plot(x,a*x+b)
    plt.show

def summary(ticker):
  data=[["P/E: ",ticker.info["trailingPE"]],["P/E expected: ",ticker.info["forwardPE"]],["market cap: ",ticker.info["marketCap"]],
   ["P/B : ",ticker.info["priceToBook"]],["current price: ",ticker.info["currentPrice"]],["D/E ratio: ",ticker.info["debtToEquity"]],
    ["Book Value: ",ticker.info["bookValue"]],["ROE: ",ticker.info["returnOnEquity"]],["Div. Yield: ",ticker.info["dividendYield"]],["cash: ",ticker.info["totalCash"]],["beta: ",ticker.info["beta"]]]
  head=["parameters","values"]
  print(tabulate(data,headers=head,tablefmt="grid"))

