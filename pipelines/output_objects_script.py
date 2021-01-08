import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from datetime import datetime, timezone, timedelta
import time

def main():
    df = pd.read_csv('data/Export_Verbruik_2010-2019_anon.csv').drop('Unnamed: 0', axis=1)
    df = df.loc[df['VERBRUIK_STARTDAT'].notna()]
    # removing object types that are not in scope
    scope = ['HHO', 'HHB', 'GZB', 'KZB', 'KZO']
    df = df.loc[df['OBJECT_TYPE_NAME'].isin(scope)]

    # we also find some rows with no water usage, so removing those too
    df = df.loc[df['VERBRUIK'] != 0]

    # coverting date-like columns to datetime
    df[['VERBRUIK_STARTDAT', 'VERBRUIK_EINDDATUM']] = df[['VERBRUIK_STARTDAT', 'VERBRUIK_EINDDATUM']].apply(pd.to_datetime)

    # calculate period and average daily consumptiopn
    df['DIFF'] = (df['VERBRUIK_EINDDATUM'] - df['VERBRUIK_STARTDAT']).dt.days
    df['AVG_DAY'] = df['VERBRUIK'] / df['DIFF']

    # grouping dataframe
    grouped = df[['OBJECT_TYPE_NAME', 'POSTCODE', 'VERBRUIK_STARTDAT', 'VERBRUIK_EINDDATUM', 'AVG_DAY']]\
    .groupby(['OBJECT_TYPE_NAME', 'POSTCODE', 'VERBRUIK_STARTDAT', 'VERBRUIK_EINDDATUM']).sum().reset_index()

    # unpivot grouped dataframe, resulting into two measurements for each row (one for start and one for end date)
    melt = grouped.reset_index().melt(id_vars=['index', 'OBJECT_TYPE_NAME', 'POSTCODE', 'AVG_DAY'], value_name='DATE').drop('variable', axis=1)
    melt['DATE'] = pd.to_datetime(melt['DATE'])

    kf = KFold(n_splits=50, shuffle=False)
   
    start = time.time()
    i = 1
    print('Starting run at: {}'.format(datetime.now()))
 
    for fold in kf.split(melt['index'].unique()):
        f = melt.loc[melt['index'].isin(fold[1])]
        # grouping by index and filling dates between start and end dates
        f = f.groupby('index').apply(lambda x: x.set_index('DATE').resample('D').first())\
                .ffill()\
                .reset_index(level=1)\
                .reset_index(drop=True)

        f.to_csv('output_objects_timeseries_{}.csv'.format(i))
        print('Completed run {} at: {}'.format(i, datetime.now()))
        i += 1
   
    print('Completed run after: {} seconds.'.format(time.time() - start))
   

try:
    main()
except Exception as err:
    print(err)