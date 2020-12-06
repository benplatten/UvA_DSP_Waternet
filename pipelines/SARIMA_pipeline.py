import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timezone, timedelta

import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf, adfuller
from statsmodels.tsa.statespace.tools import diff
from statsmodels.tsa.seasonal import seasonal_decompose, STL

from pmdarima import auto_arima
from scipy.stats import norm
from calendar import monthrange

# Custom transformer import
import os, sys
sys.path.append(os.getcwd())
from custom_modules.custom_transformers import TimeSeriesTransformer

def prepocess(data):
    data['Datetime_UTC'] = data['Datetime_UTC'].apply(lambda x: pd.to_datetime(datetime.fromisoformat(x.replace("Z", "+00:00")), unit='ms'))
    data = data.loc[data.Datetime_UTC.between('2010-01-01', '2020-08-31')]
    data.set_index('Datetime_UTC', inplace=True)
    data = data.resample('d').sum().fillna(0) / 60
    data['day_of_yr'] = pd.Series(data.index.date).apply(lambda x: x.timetuple().tm_yday).tolist()
    return data

def trim_outliers(df, value, grouper, q1=0.25, q3=0.75, iqr=1.5):
    # Function to filter outliers iteratively, based on boxplots and IQR
    n = 1
    
    while n > 0:
        df_grouped = df.groupby(df[grouper])[value]
        Q1 = df_grouped.quantile(q1)
        Q3 = df_grouped.quantile(q3)
        IQR = iqr * (Q3 - Q1)

        df = df.drop('min_whis', axis=1, errors='ignore').join((Q1 - IQR).rolling(3, min_periods=1).mean().rename(\
            'min_whis'), on='day_of_yr') 
        df = df.drop('max_whis', axis=1, errors='ignore').join((Q3 + IQR).rolling(3, min_periods=1).mean().rename(\
            'max_whis'), on='day_of_yr') 

        n = sum((df[value] < df['min_whis']) | (df[value] > df['max_whis']))
        print('number of outliers to trim: {}'.format(n))

        df[value].mask( (df[value] < df['min_whis']) | (df[value] > df['max_whis']), inplace=True)
        df[value].interpolate(method='linear', inplace=True)
    return df

def SARIMAX(y):
    mod = sm.tsa.statespace.SARIMAX(y,
                                order=(0, 0, 3),
                                seasonal_order=(0, 1, 1, 12),
                                trend='n',
                                mle_regression=True,
                                enforce_stationarity=True,
                                enforce_invertibility=True)
    return mod.fit()
    

def pipeline():
    columns = ['Datetime_UTC', 'Demand_m3', 'Data_status']
    data = pd.read_csv('../data/Water_usage_minute_2010_2020.txt', names=columns, header=None, delimiter='|')
    data = prepocess(data)
    data = trim_outliers(data, 'Demand_m3', 'day_of_yr')

    y = pd.DataFrame(data.resample('M').sum()['Demand_m3'])
    # Setting train and test sets
    train, test = y.loc['2010-1-1':'2018-12-31'], y.loc['2019-1-1'::]
    
    result = STL(train, period=12).fit()
    # deflate series
    train['trend'] = result.trend
    train['trf_demand'] = train['Demand_m3'] - train['trend']
    # initialize a transformer class
    TT = TimeSeriesTransformer(log=False, detrend=False, diff=True, scale=False)
    # detrend and perform differencing on both 1 and 12 periods
    train['trf_demand'] = TT.fit_transform(train.index, train.trf_demand, [1])

    y_train = train['trf_demand'][1::]
    y_test = test['Demand_m3']

    # train model
    results = SARIMAX(y_train)

    # get forecast
    pred_uc = results.get_forecast(steps=y_test.shape[0])
    df_pred = pred_uc.summary_frame().rename(columns={'mean': 'y_hat', 'mean_se': 'y_hat_se'})

    # append df_pred to data
    df_com = train.append(test)
    df_com = df_com.join(df_pred, how='left')
    
    # extrapolate trendline
    OLS = TT.get_trend(df_com.trend['2017':'2018'])
    last = df_com.loc[df_com.trend.notna()].trend.tolist()[-1]
    extrap = df_com.loc[df_com.trend.isna()].trend.tolist()

    for i, x in enumerate(extrap):
        extrap[i] = last + (i + 1) * OLS[1][0]
    
    # reconstruct dataframe to include forecast
    df_com.trend = df_com.loc[df_com.trend.notna()].trend.tolist() + extrap
    df_com['y_pred'] = TT.start[1].append(df_com.y_hat.fillna(df_com.trf_demand)[1::]).cumsum() + df_com.trend
    df_com['ci_lower'] = df_com.y_pred - abs(df_com.y_hat - df_com.mean_ci_lower)
    df_com['ci_upper'] = df_com.y_pred + abs(df_com.mean_ci_upper - df_com.y_hat)
    df_com.loc[df_com.y_hat.isna(), ['y_pred', 'ci_lower', 'ci_upper']] = np.nan

    ax = df_com.Demand_m3.plot(label='observed')
    df_com.y_pred.plot(ax=ax, label='forecast', alpha=.7, figsize=(14, 4))

    ax.fill_between(df_com.index,
                    df_com.ci_lower,
                    df_com.ci_upper, color='k', alpha=.2)
    ax.set_xlabel('Date')
    ax.set_ylabel('Demand_m3')

    plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
    plt.show()


##############################################################
###                     Main function call                 ###
##############################################################
if __name__ == '__main__':
    try:
        pipeline()
    except Exception as ex:
        print(ex)