#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 00:37:57 2020

@author: TG
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from textwrap import wrap


class Index_Replication:
    
    def __init__(self, df, n, h, l):
        self.df = df
        self.n = n
        self.h = h
        self.l = l
        
    def st_df(self): #standardize the stock returns
        df = self.df
        me = df.mean()
        stdev = (df - me).std()
        st_df = df / stdev
        return st_df
        
    # def st_df(self):
    #     return self.df
    
    def cov(self):
        df = self.st_df()
        cov = df.cov()
        return cov
    
    
    def pca(self):
        n = self.n
        df = self.cov()
        pca = PCA(n_components=n)
        pca.fit(df)
        return pca
    
    def z(self):
        st_df = self.st_df()
        pca = self.pca()
        U = pca.components_
        z = np.dot(st_df, U.T)
        return z
    
    def x_prime(self):
        pca = self.pca()
        U = pca.components_
        z = self.z()
        return np.dot(U.T, z.T)
    
    def d(self):
        st_df = self.st_df()
        a = pd.DataFrame(data=self.x_prime()).T
        a.columns = st_df.columns
        a.index = st_df.index
        d = ((a - st_df) ** 2).mean()
        return d
        
    def sort(self):
        d = self.d()
        sort = d.sort_values()
        return sort
    
    def select(self):
        df = self.df
        h = self.h
        l = self.l
        sort = self.sort()
        small_d = sort[:h]
        large_d = sort[-l:]
        sel = df[small_d.append(large_d).index]
        return sel
    
    def weights(self, stocks, market):
        h = self.h                     #short is not allowed
        l = self.l
        w0 = np.array([1/(h+l) for i in range(h+l)])
        res = minimize(f(stocks, market), w0, method = 'SLSQP', \
                        constraints = ({'type': 'ineq', 'fun': lambda x: x}, \
                                      {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}), \
                                          tol=0.000000000000000001)
        return res.x
        
#         h = self.h                         #short is allowed
#         l = self.l
#         w0 = np.array([1/(h+l) for i in range(h+l)])
#         res = minimize(f(stocks, market), w0, method = 'SLSQP', \
#                         constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}), \
#                                           tol=0.000000000000000001)
#         return res.x

    
def tracking_errors(w, stocks, market):
    w = np.matrix(w).T
    stock_ret = stocks * w
#     d = np.sqrt(np.sum(np.array((market - stock_ret)) ** 2)) / len(stock_ret)
    d = np.std(market - stock_ret)
    return d

def f(stocks, market):
    v = lambda x: tracking_errors(x, stocks, market)
    return v


###########################################################    
class Adjust_Weight: #Adjusting weights
    
    def __init__(self, df, df_ndx, n, h, l):
        self.df = df
        self.df_ndx = df_ndx
        self.n = n
        self.h = h
        self.l = l
    
    def first(self):
        df = self.df
        df_ndx = self.df_ndx
        df = df.loc[:'12/31/18']
        df_ndx = df_ndx.loc[:'12/31/18']
        r = Index_Replication(df, self.n, self.h, self.l)
        select = r.select()
        stocks = np.matrix(select)
        market = np.array(df_ndx)
        return np.array(select.columns), r.weights(stocks, market)
    
    def after_first(self):
        first = self.first()           #data are only added(not rolling)
        weights = pd.DataFrame(index=range(15), columns=first[0])
        weights.iloc[0] = first[1]
        
        for i in range(13):
            start = 501 + i * 22
            df = self.df
            df_ndx = self.df_ndx
            df = df.iloc[:start+22]
            df_ndx = df_ndx.iloc[:start+22]
            r = Index_Replication(df, self.n, self.h, self.l)
            select = r.select()
            stocks = np.matrix(select)
            market = np.array(df_ndx)
            weights.iloc[i+1] = r.weights(stocks, market)
            
        start = 501 + 13 * 22
        df = self.df
        df_ndx = self.df_ndx
        df = df.iloc[:start+23]
        df_ndx = df_ndx.iloc[:start+23]
        r = Index_Replication(df, self.n, self.h, self.l)
        select = r.select()
        stocks = np.matrix(select)
        market = np.array(df_ndx)
        weights.iloc[14] = r.weights(stocks, market)
        return weights
        
        ##################################################################
        
        # first = self.first()         #two-year rolling
        # weights = pd.DataFrame(index=range(15), columns=first[0])
        # weights.iloc[0] = first[1]
        
        # for i in range(13):
        #     start = 501 + i * 22
        #     df = self.df
        #     df_ndx = self.df_ndx
        #     df = df.iloc[(i+1)*22:start+22]
        #     df_ndx = df_ndx.iloc[(i+1)*22:start+22]
        #     r = Index_Replication(df, self.n, self.h, self.l)
        #     select = r.select()
        #     stocks = np.matrix(select)
        #     market = np.array(df_ndx)
        #     weights.iloc[i+1] = r.weights(stocks, market)
            
        # start = 501 + 13 * 22
        # df = self.df
        # df_ndx = self.df_ndx
        # df = df.iloc[14*22:start+23]
        # df_ndx = df_ndx.iloc[14*22:start+23]
        # r = Index_Replication(df, self.n, self.h, self.l)
        # select = r.select()
        # stocks = np.matrix(select)
        # market = np.array(df_ndx)
        # weights.iloc[14] = r.weights(stocks, market)
        # return weights
        
        ####################################################################
    
        # first = self.first()         #one-month rolling
        # weights = pd.DataFrame(index=range(15), columns=first[0])
        # weights.iloc[0] = first[1]
        
        # for i in range(13):
        #     start = 501 + i * 22
        #     df = self.df
        #     df_ndx = self.df_ndx
        #     df = df.iloc[start:start+22]
        #     df_ndx = df_ndx.iloc[start:start+22]
        #     r = Index_Replication(df, self.n, self.h, self.l)
        #     select = r.select()
        #     stocks = np.matrix(select)
        #     market = np.array(df_ndx)
        #     weights.iloc[i+1] = r.weights(stocks, market)
            
        # start = 501 + 13 * 22
        # df = self.df
        # df_ndx = self.df_ndx
        # df = df.iloc[start:start+23]
        # df_ndx = df_ndx.iloc[start:start+23]
        # r = Index_Replication(df, self.n, self.h, self.l)
        # select = r.select()
        # stocks = np.matrix(select)
        # market = np.array(df_ndx)
        # weights.iloc[14] = r.weights(stocks, market)
        # return weights
    
    def backtest(self):
        df = self.df
        df = df.iloc[501:]
        df_ndx = self.df_ndx
        df_ndx = df_ndx.iloc[501:]
        tickers = self.first()[0]
        df = df[tickers]
        df = np.matrix(df)
        weights = self.after_first()
        weights = np.matrix(weights.T)
        product = df * weights
        returns = np.zeros(331)
        for i in range(14):
            returns[i*22:(i+1)*22] = product[i*22:(i+1)*22, i].T
        returns[14*22:] = product[14*22:, 14].T
        nsdq = np.array(df_ndx.T)[0]
#         TE = np.sqrt(np.sum((returns - nsdq) ** 2)) / len(returns)
        TE = np.std(returns - nsdq)
        return np.cumprod(returns + 1), np.cumprod(nsdq + 1), TE
    
    def plot_weights(self):
        h = self.h
        l = self.l
        df = self.df
        df = df.iloc[501:]
        ind = df.index
        a = np.zeros(15).astype('str')
        for i in range(14):
            a[i] = '%s - %s' % (ind[i*22], ind[(i+1)*22-1])
        a[14] = '%s - %s' % (ind[14*22], ind[-1])
        a = ['\n'.join(wrap(l, 3)) for l in a]
        af = self.after_first()
        weights = np.array(af)
        plt.figure()
        for i in range(h+l):
            plt.plot(a, weights[:, i], label=af.columns[i])
        plt.title('Weights of the Stocks During Each Time Period')
        plt.xlabel('Time Period')
        plt.ylabel('Weight')
        plt.legend(bbox_to_anchor=(1.05, 1))
        plt.show()
        
    def plot_return(self):
        xs = np.arange(1, 332)
        bt = self.backtest()
        y1 = bt[0]
        y2 = bt[1]
        plt.figure()
        plt.plot(xs, y1, label='My Index Replication')
        plt.plot(xs, y2, label='Nasdaq Index')
        plt.title('Cumulative Return over Time')
        plt.xlabel('Day')
        plt.ylabel('Return')
        plt.legend(bbox_to_anchor=(1.05, 1))
        plt.show()
    

if __name__ == '__main__':
    #setting initial values
    filename = 'data.csv'
    filename_ndx = 'nasdaq.csv'
    df = pd.read_csv(filename)
    df_ndx = pd.read_csv(filename_ndx)
    df.index = df['Date']
    df = df.iloc[:, 1:]
    df_ndx.index = df_ndx['Date']
    df_ndx = df_ndx.iloc[:, 1:]
    col_na = df.columns[df.isna().any()].tolist()
    df = df.drop(col_na, axis=1)
    
    df = (df - df.shift(1)) / df
    df = df.iloc[1:]
    df_ndx = (df_ndx - df_ndx.shift(1)) / df_ndx
    df_ndx = df_ndx.iloc[1:]
    
    for i in range(1, 4):
       n = i
       print('\nWhen n = %d:' % n, \
             '\nh is the number of stocks with more common information with the market')
       table = pd.DataFrame(columns=['Cumulative Difference', 'Tracking Error'])
    
       for h in range(1, 15):
           l = 15 - h
           a = Adjust_Weight(df, df_ndx, n, h, l)
           bt = a.backtest()
           a0 = bt[0]
           a1 = bt[1]
           a2 = bt[2]
           st = 'h = ' + str(h)
           table.loc[st] = [round(abs(a0[-1] - a1[-1]), 4), round(a2, 6)]
    
       print(table, '\n')
       
    #     n is the number of principal components that we are gonna use
    #     h is the number of stocks having more common information with the market
    #     l is the number of stocks having less common information with the market
    
    h = np.array([6, 7, 8, 12, 5, 6, 7, 8])    
    l = 15 - h
    n = np.array([1, 1, 1, 1, 2, 2, 2, 2])
    
    
    for i in range(len(h)):
    
        a = Adjust_Weight(df, df_ndx, n[i], h[i], l[i])
        bt = a.backtest()
        a0 = bt[0]
        a1 = bt[1]
        a2 = bt[2]
    
        print('\nWhen n=%d, h=%d, l=%d:' % (n[i], h[i], l[i]))
        a.plot_return()
        a.plot_weights()
    
        print('The cumulative return of my index replication is %.2f' % \
              (100 * (a0[-1] - 1)) + '%')
        print('The cumulative return of the Nasdaq index is %.2f' % \
              (100 * (a1[-1] - 1)) + '%')    
        print('The Tracking Error of my index replication is: %.6f\n' \
              % a2)
    
    
    
    
    






