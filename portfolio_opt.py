from pandas_datareader import data as web #to access data from yahoo finance
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns

assets =  ["FB", "AMZN", "AAPL", "NFLX", "GOOG"]
#this is a list of ticker symbols for portfolio

weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
#self-assigned weights

stockStartDate = '2013-01-01'
today = datetime.today().strftime('%Y-%m-%d')
#create start date and end date which is current date

df = pd.DataFrame()
#create a dataframe

for stock in assets:
   df[stock] = web.DataReader(stock,data_source='yahoo',start=stockStartDate , end=today)['Adj Close']
#to access data from yahoo finance

#insert adj-close prices into graph
'''title = 'Portfolio Adj. Close Price History    '
#Get the stocks
my_stocks = df
#Create and plot the graph
plt.figure(figsize=(12.2,4.5)) #width = 12.2in, height = 4.5
# Loop through each stock and plot the Adj Close for each day
for c in my_stocks.columns.values:
  plt.plot( my_stocks[c],  label=c)#plt.plot( X-Axis , Y-Axis, line_width, alpha_for_blending,  label)
plt.title(title)
plt.xlabel('Date',fontsize=18)
plt.ylabel('Adj. Price USD ($)',fontsize=18)
plt.legend(my_stocks.columns.values, loc='upper left') 
'''
returns = df.pct_change()
#calculate precent changes

cov_matrix_annual = returns.cov() * 252
#create a variance for each column then calculate covariance between each column including itself (for stock year)

port_variance = np.dot(weights.T, np.dot(cov_matrix_annual, weights))
#formula used for portfolio variance
port_volatility = np.sqrt(port_variance)
#square root is used to get portfolio standard deviation
portfolioSimpleAnnualReturn = np.sum(returns.mean()*weights) * 252
#get average of each stock, then weigh it, add it all together then adjust for stock year

mu = expected_returns.mean_historical_return(df)
#returns.mean() * 252

S = risk_models.sample_cov(df) #Get the sample covariance matrix

ef = EfficientFrontier(mu, S)
weights = ef.max_sharpe()
#Maximize the Sharpe ratio, and get the raw weights

cleaned_weights = ef.clean_weights()
#rounds weights for each stock

print(cleaned_weights)
ef.portfolio_performance(verbose=True)
#gives three measurements for the portfolio

from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
latest_prices = get_latest_prices(df)
print(latest_prices)#get and print latest stock price



weights = cleaned_weights 
da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=15000)
#get allocation for portfolio stocks and calculate leftover money
allocation, leftover = da.lp_portfolio()
print("Discrete allocation:", allocation)
print("Funds remaining: ${:.2f}".format(leftover))