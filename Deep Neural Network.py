# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 11:52:02 2020

@author: Peicheng Wang  & Zehao Dong
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import yfinance as yf

from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
from keras.models import load_model
from sklearn.preprocessing import StandardScaler  
from collections import defaultdict
from sklearn.decomposition import PCA
from scipy.optimize import minimize


plt.style.use('seaborn')

def read_data(stockt, indext, starting, ending):
    start = starting
    end = ending
    data1 = yf.download(stockt, start = start, end = end)
    data2 = yf.download(indext, start = start, end = end)
    Adj_close1 = data1["Adj Close"].dropna(axis = 0, how = "all").dropna(axis = 1, how = "any")
    Adj_close2 = data2["Adj Close"].dropna()
    return Adj_close1, Adj_close2 
    
def standardization(df):
    mu = df.mean()
    sigma = df.std()
    result = (df - mu)/sigma
    return result

def autoencoder(df):
    input_size = df.shape[1]
    hidden_size1 = int(input_size/6) 
    hidden_size2 = int(hidden_size1/3)
    
    x = Input(shape=(input_size,))
    
    ## Encoder 2 layers
    h1 = Dense(hidden_size1, activation='tanh')(x)
    z = Dense(hidden_size2, activation='tanh')(h1)
    
    ## Decoder 2 layers
    h2 = Dense(hidden_size1, activation='tanh')(z)
    y = Dense(input_size, activation='tanh')(h2)
      
    autoencoder = Model(input=x, output=y)
    autoencoder.compile(optimizer='sgd', loss='mse')
    
    autoencoder.fit(df, df, epochs=500, batch_size=20, verbose=0)

    return autoencoder

def get_normlist(df, target):
    result = (df - target).apply(lambda x: np.sqrt(sum(x **2)))
    return result 

def plot_comparison(sortedlist, order, index10, stock10, phase):
    stock11 = stock10[phase]["sprice"][sortedlist.index[order]]
    index11 = index10[phase]["sprice"]
    plt.plot(stock11, label = sortedlist.index[order])
    plt.plot(index11, label = "index")
    plt.title('{} vs index'.format(sortedlist.index[order]))
    plt.legend()
    plt.show()
    
def get_weight(data1, data2, factor):
    """
    hidden layer and output layer are both linear
    """
    inputdata = data1.loc[:, factor]
    outputdata = data2
    input_size = inputdata.shape[1]
    hidden_size = int(input_size / 2)
    output_size = 1

    x1 = Input(shape=(input_size,))

    h = Dense(hidden_size, activation='linear')(x1)
    y = Dense(output_size, activation='linear')(h)

    model_ = Model(input=x1, output=y)
    model_.compile(optimizer='sgd', loss='mse')

    model_.fit(inputdata, outputdata,
               epochs=50,
               batch_size=5,
               verbose=0)
    w = model_.layers[1].get_weights()[0]
    v = model_.layers[2].get_weights()[0]
    weight = (np.dot(w, v)).flatten()
    weight = weight / np.std(inputdata.values)
    weight = weight / sum(weight)
    return weight
    
def rebalance_weight(pos, factor1, stock1, index1):
    last_index = pos.index[-1]
    last_loc = stock1.index.get_loc(last_index)
    df = stock1.iloc[last_loc - period:last_loc, :]
    y = index1.iloc[last_loc - period:last_loc]
    weight = get_weight(df, y, factor1)
    return weight   

    
def rebalance_weight2(pos, factor1, stock1, index1):
    last_index = pos.index[-1]
    last_loc = stock1.index.get_loc(last_index)
    df = stock1.iloc[last_loc - period:last_loc, :]
    y = index1.iloc[last_loc - period:last_loc]
    weight = get_weight2(df, y, factor1)
    return weight  

def fun(arr, weight, factor1):
    a = weight[str(arr.name)[:7]]
    return np.dot(arr[factor1], a)

def get_TE(arr1, arr2):
    te = np.std(arr1-arr2)
    return te

def pca(n, df):
    cov = df.cov()
    pca = PCA(n_components=n)
    pca.fit(cov)
    U = pca.components_
    z = np.dot(df, U.T)
    res = np.dot(U.T, z.T)
    return res.T

def get_weight2(data1, data2, factor):
    inputdata =data1.loc[:, factor]
    outputdata = data2

    def f(w, stocks, market):
        v = np.std(np.dot(stocks, w) - market)
        return v

    w0 = np.ones(15) / 15
    args = (inputdata, outputdata)
    res = minimize(f, w0, args=args, method='SLSQP', \
                   constraints=({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}), \
                   tol=0.000000000000000001).x

    return res

###################################################################
## S&P 500 Index Replication
## read index compoments
DATA = pd.read_csv('tickers.csv')
ticker_stock = list(DATA.iloc[:,0])
ticker_index = ["^GSPC"]

## time period
start = "2017-01-01"
end = "2020-04-20"

## read data
temp = read_data(ticker_stock, ticker_index, start, end)
raw_stock = temp[0]
raw_index = temp[1]

## save data
stock = defaultdict(defaultdict)
index = defaultdict(defaultdict)    

## standardized train data
stock["train"]["price"] = raw_stock.iloc[:-200,:]
stock["train"]["sprice"] = standardization(stock["train"]["price"])
stock["train"]["pg"] = np.log(raw_stock.iloc[:-200,:].pct_change().dropna(axis = 0, how = 'all') + 1)
stock["train"]["spg"] = standardization(stock["train"]["pg"])
index["train"]["price"] = raw_index.iloc[:-200]
index["train"]["sprice"] = standardization(index["train"]["price"])
index["train"]["pg"] = np.log(raw_index.iloc[:-200].pct_change().dropna(axis = 0, how = 'all') + 1)
index["train"]["spg"] = standardization(index["train"]["pg"])

## standardize verify data
stock["verify"]["price"] = raw_stock.iloc[-326:]
stock["verify"]["sprice"] = standardization(stock["verify"]["price"])
stock["verify"]["pg"] = np.log(raw_stock.iloc[-326:].pct_change().dropna(axis = 0, how = 'all') + 1)
stock["verify"]["spg"] = standardization(stock["verify"]["pg"])
index["verify"]["price"] = raw_index.iloc[-326:]
index["verify"]["sprice"] = standardization(index["verify"]["price"])
index["verify"]["pg"] = np.log(raw_index.iloc[-326:].pct_change().dropna(axis = 0, how = 'all') + 1)
index["verify"]["spg"] = standardization(index["verify"]["pg"])

## train model
model = autoencoder(stock["train"]["sprice"])
predict = model.predict(stock["train"]["sprice"])
norm_list = get_normlist(stock["train"]["sprice"], predict)
sorted_norm_list = norm_list.sort_values()

## plot to verify stock selection
plot_comparison(sorted_norm_list, 0, index, stock, "train")
plot_comparison(sorted_norm_list, -1, index, stock, "train")

## in-sample training
factor = [sorted_norm_list.index[0],
          sorted_norm_list.index[1],
          sorted_norm_list.index[2],
          sorted_norm_list.index[3],
          sorted_norm_list.index[-11],
          sorted_norm_list.index[-10],
          sorted_norm_list.index[-9],
          sorted_norm_list.index[-8],
          sorted_norm_list.index[-7],
          sorted_norm_list.index[-6],
          sorted_norm_list.index[-5],
          sorted_norm_list.index[-4],
          sorted_norm_list.index[-3],
          sorted_norm_list.index[-2],
          sorted_norm_list.index[-1]]

weight = get_weight(stock['train']['spg'], index['train']['spg'], factor)

## plot trainning comparison
plt.figure()
plt.plot(pd.DataFrame(np.dot(stock['train']['pg'][factor], weight).cumsum(), index=stock['train']['pg'].index))
plt.plot(index['train']['pg'].cumsum())
plt.show()

## TE
print("TE", get_TE(np.dot(stock['train']['pg'][factor], weight), index['train']['pg']))


## out of sample test
period = 21*6

## caculate the array of weight allocation
weight_df = stock["verify"]["spg"].iloc[period:,1].resample("BM").apply(rebalance_weight, factor1 = factor, index1 = index["verify"]["spg"], 
                  stock1 = stock["verify"]["spg"])
weight_df.index=weight_df.index.astype(str).str[:7]
weight_df=weight_df.apply(lambda x: x/sum(x))
new_weight_df=weight_df.shift().dropna()

## caculate the rebalance return
r = stock['verify']['pg'].iloc[period + 21:].apply(fun, weight = new_weight_df, factor1 = factor, axis = 1)
plt.figure()
plt.plot(r.cumsum())
plt.plot(index['verify']['pg'][period + 21:].cumsum())
plt.show()


print('TE:',  get_TE(r.values, index['verify']['pg'][period + 21:].values))

## SPY TE
start1 = "2019-01-02"
end1 = "2020-04-20"
spy = yf.download("SPY", start = start1, end = end1)["Adj Close"].dropna()
ret_spy = np.log(spy.pct_change().dropna() + 1)
print("SPY", get_TE(ret_spy[period + 21:], index['verify']['pg'][period + 21:].values))

## print out all possible portfolio results
number_top = np.zeros(16)
TE_list = np.zeros(16)
cumulative_list = np.zeros(16)
cumulative_error = np.zeros(16)
for i in range(16):
    factor_temp = [
          sorted_norm_list.index[i-15],
          sorted_norm_list.index[i-14],
          sorted_norm_list.index[i-13],
          sorted_norm_list.index[i-12],
          sorted_norm_list.index[i-11],
          sorted_norm_list.index[i-10],
          sorted_norm_list.index[i-9],
          sorted_norm_list.index[i-8],
          sorted_norm_list.index[i-7],
          sorted_norm_list.index[i-6],
          sorted_norm_list.index[i-5],
          sorted_norm_list.index[i-4],
          sorted_norm_list.index[i-3],
          sorted_norm_list.index[i-2],
          sorted_norm_list.index[i-1],]
    
    period = 21*6

    ## caculate the array of weight allocation
    weight_df_temp = stock["verify"]["spg"].iloc[period:,1].resample("BM").apply(rebalance_weight, factor1 = factor_temp, index1 = index["verify"]["spg"], 
        stock1 = stock["verify"]["spg"])
    weight_df_temp.index=weight_df_temp.index.astype(str).str[:7]
    weight_df_temp=weight_df_temp.apply(lambda x: x/sum(x))
    new_weight_df_temp=weight_df_temp.shift().dropna()
        
        ## caculate the rebalance return
    r_temp = stock['verify']['pg'].iloc[period + 21:].apply(fun, weight = new_weight_df_temp, factor1 = factor_temp, axis = 1)
    plt.figure()
    plt.plot(r_temp.cumsum(), label = "Replicating Portfolio Return")
    plt.plot(index['verify']['pg'][period + 21:].cumsum(), label = "Index Return")
    plt.plot(ret_spy[period + 21:].cumsum(), label = "SPY Return")
    plt.title('top {} stocks portfolio vs index'.format(i))
    plt.legend()
    plt.show()
        
        
    TE_temp = get_TE(r_temp.values, index['verify']['pg'][period + 21:].values)
    number_top[i] = i
    TE_list[i] = TE_temp
    cumulative_list[i] = r_temp.cumsum()[-1]
    cumulative_error[i] = r_temp.cumsum()[-1] - index['verify']['pg'][period + 21:].cumsum()[-1] 
 
## print the results      
report = {'number of top': number_top, 'Tracking Error': TE_list, 'cumulative return': 
    cumulative_list, 'cumulative return error':cumulative_error} 
report = pd.DataFrame(report)
report    

############################################################
## Nasdaq Index Replication
raw_stock1 = pd.read_csv("data.csv", parse_dates=True, index_col=0)
raw_index1 =  yf.download("^NDX", start = "2017-01-01", end = "2020-04-25")["Adj Close"]

## save data
stock1 = defaultdict(defaultdict)
index1 = defaultdict(defaultdict)

## standardized train data
stock1["train"]["price"] = raw_stock1.loc[:"2018-12-31",:]
stock1["train"]["sprice"] = standardization(stock1["train"]["price"])
stock1["train"]["pg"] = np.log(raw_stock1.loc[:"2018-12-31",:].pct_change().dropna(axis = 0, how = 'all') + 1)
stock1["train"]["spg"] = standardization(stock1["train"]["pg"])
index1["train"]["price"] = raw_index1.loc[:"2018-12-31"]
index1["train"]["sprice"] = standardization(index1["train"]["price"])
index1["train"]["pg"] = np.log(raw_index1.loc[:"2018-12-31"].pct_change().dropna(axis = 0, how = 'all') + 1)
index1["train"]["spg"] = standardization(index1["train"]["pg"])

## standardize verify data
stock1["verify"]["price"] = raw_stock1.iloc[501 - 126:,:]
stock1["verify"]["sprice"] = standardization(stock1["verify"]["price"])
stock1["verify"]["pg"] = np.log(raw_stock1.iloc[501 - 126:,:].pct_change().dropna(axis = 0, how = 'all') + 1)
stock1["verify"]["spg"] = standardization(stock1["verify"]["pg"])
index1["verify"]["price"] = raw_index1.iloc[501 - 126:]
index1["verify"]["sprice"] = standardization(index1["verify"]["price"])
index1["verify"]["pg"] = np.log(raw_index1.iloc[501 - 126:].pct_change().dropna(axis = 0, how = 'all') + 1)
index1["verify"]["spg"] = standardization(index1["verify"]["pg"])

## train model
model1 = autoencoder(stock1["train"]["sprice"])
predict1 = model1.predict(stock1["train"]["sprice"])
norm_list1 = get_normlist(stock1["train"]["sprice"], predict1)
sorted_norm_list1 = norm_list1.sort_values()

## training portfolio
factor1 = [sorted_norm_list1.index[0],
          sorted_norm_list1.index[1],
          sorted_norm_list1.index[2],
          sorted_norm_list1.index[3],
          sorted_norm_list1.index[-11],
          sorted_norm_list1.index[-10],
          sorted_norm_list1.index[-9],
          sorted_norm_list1.index[-8],
          sorted_norm_list1.index[-7],
          sorted_norm_list1.index[-6],
          sorted_norm_list1.index[-5],
          sorted_norm_list1.index[-4],
          sorted_norm_list1.index[-3],
          sorted_norm_list1.index[-2],
          sorted_norm_list1.index[-1]]
weight1 = get_weight(stock1['train']['spg'], index1['train']['spg'], factor1)

## plot trainning comparison
plt.figure()
plt.plot(pd.DataFrame(np.dot(stock1['train']['pg'][factor1], weight1).cumsum(), index=stock1['train']['pg'].index), label = "Portfolio")
plt.plot(index1['train']['pg'].cumsum(), label = "index")
plt.legend()
plt.show()

## TE
print("TE", get_TE(np.dot(stock1['train']['pg'][factor1], weight1), index1['train']['pg']))

## QQQ ETF TE
start2 = "2017-01-01"
end2 = "2019-01-01"
nsa1 = yf.download("QQQ", start = start2, end = end2)["Adj Close"].dropna()
ret_nsa1 = np.log(nsa1.pct_change().dropna() + 1)
print("QQQ", get_TE(ret_nsa1, index1['train']['pg'].values))

## Nasdaq
start3 = "2018-06-29"
nsa2 = yf.download("QQQ", start = start3)["Adj Close"].dropna()
ret_nsa2 = np.log(nsa2.pct_change().dropna() + 1)

weight1 = defaultdict(defaultdict)

#### Out of Sample Rebalancing Backtest
number_top1 = np.zeros(16)
TE_list1 = np.zeros(16)
cumulative_list1 = np.zeros(16)
cumulative_error1 = np.zeros(16)
for i in range(16):
    factor_temp1 = [
          sorted_norm_list1.index[i-15],
          sorted_norm_list1.index[i-14],
          sorted_norm_list1.index[i-13],
          sorted_norm_list1.index[i-12],
          sorted_norm_list1.index[i-11],
          sorted_norm_list1.index[i-10],
          sorted_norm_list1.index[i-9],
          sorted_norm_list1.index[i-8],
          sorted_norm_list1.index[i-7],
          sorted_norm_list1.index[i-6],
          sorted_norm_list1.index[i-5],
          sorted_norm_list1.index[i-4],
          sorted_norm_list1.index[i-3],
          sorted_norm_list1.index[i-2],
          sorted_norm_list1.index[i-1],]
    
    period = 21*6

    ## caculate the array of weight allocation
    weight_df_temp1 = stock1["verify"]["spg"].iloc[period:,1].resample("BM").apply(rebalance_weight, factor1 = factor_temp1, index1 = index1["verify"]["spg"], 
        stock1 = stock1["verify"]["spg"])
    weight_df_temp1.index=weight_df_temp1.index.astype(str).str[:7]
    weight_df_temp1=weight_df_temp1.apply(lambda x: x/sum(x))
    new_weight_df_temp1=weight_df_temp1.shift().dropna()
        
    ## caculate the rebalance return
    r_temp1 = stock1['verify']['pg'].iloc[period + 21:].apply(fun, weight = new_weight_df_temp1, factor1 = factor_temp1, axis = 1)
    plt.figure()
    plt.plot(r_temp1.cumsum(), label = "Replicating Portfolio Return")
    plt.plot(index1['verify']['pg'][period + 21:].cumsum(), label = "index")
    plt.plot(ret_nsa2[period + 21:].cumsum(), label = "QQQ Return")
    plt.title('top {} stocks portfolio vs index'.format(i))
    plt.legend()
    plt.show()
        
    weight1[str(i) + "top weight"] = new_weight_df_temp1
    
    TE_temp1 = get_TE(r_temp1.values, index1['verify']['pg'][period + 21:].values)
    number_top1[i] = i
    TE_list1[i] = TE_temp1
    cumulative_list1[i] = r_temp1.cumsum()[-1]
    cumulative_error1[i] = r_temp1.cumsum()[-1] - index1['verify']['pg'][period + 21:].cumsum()[-1] 
       
    
report1 = {'number of top': number_top1, 'Tracking Error': TE_list1, 'cumulative return': 
    cumulative_list1, 'cumulative return error':cumulative_error1} 
report1 = pd.DataFrame(report1)
report1 