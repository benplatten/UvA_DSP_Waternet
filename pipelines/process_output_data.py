import pandas as pd
import numpy as np
import logging

def process_output_data(output_data,postcode_dict):

    df = output_data

    df.drop(['Unnamed: 0'],axis=1,inplace=True)

    columns_eng = ["CONSUMPTION_ID","CONSUMPTION_OBJECT_ID","CONSUMPTION_START_DATE",
               "CONSUMPTION_END_DATE","CONSUMPTION_ESTRATED_YN","CONSUMPTION","POSTCODE","CITY","OBJECT_TYPE_NAME"]

    df.columns = columns_eng
    df['CONSUMPTION_START_DATE'] = df['CONSUMPTION_START_DATE'].apply(lambda x: pd.to_datetime(x, format='%Y-%m-%d'))
    df['CONSUMPTION_END_DATE'] = df['CONSUMPTION_END_DATE'].apply(lambda x: pd.to_datetime(x, format='%Y-%m-%d'))
    
    df['region'] = df['POSTCODE'].map(postcode_dict) 
    df.dropna(subset=['region'],inplace=True)

    df['period'] = df['CONSUMPTION_END_DATE'] - df['CONSUMPTION_START_DATE']
    df['period'] = df['period'].dt.days
    df['consumption_per_day'] = df['CONSUMPTION'] / df['period']

    df1 = pd.concat([pd.Series(r.CONSUMPTION_ID,pd.date_range(r.CONSUMPTION_START_DATE, r.CONSUMPTION_END_DATE, freq='YS')) 
                 for r in df.itertuples()]).reset_index()
    df1.columns = ['CONSUMPTION_START_DATE','CONSUMPTION_ID']

    df2 = (pd.concat([df[['CONSUMPTION_ID','CONSUMPTION_START_DATE']], df1], sort=False, ignore_index=True)
         .sort_values(['CONSUMPTION_ID','CONSUMPTION_START_DATE'])
         .reset_index(drop=True))

    mask = df2['CONSUMPTION_ID'].duplicated(keep='last')
    s = df2['CONSUMPTION_ID'].map(df.set_index('CONSUMPTION_ID')['CONSUMPTION_END_DATE'])
    df2['CONSUMPTION_END_DATE'] = np.where(mask, df2['CONSUMPTION_START_DATE'] + pd.offsets.YearEnd(), s)

    df_values = df[['CONSUMPTION_ID','CONSUMPTION_OBJECT_ID','consumption_per_day']]
    df_categories = df.drop(['CONSUMPTION_START_DATE','CONSUMPTION_END_DATE','period','CONSUMPTION','consumption_per_day','CONSUMPTION_ID'],axis=1)
    df_categories = df_categories.groupby(['CONSUMPTION_OBJECT_ID']).first().reset_index()

    consumption_yearly = df2.merge(df_values,on='CONSUMPTION_ID',how='left')
    consumption_yearly['days'] = consumption_yearly['CONSUMPTION_END_DATE'] - consumption_yearly['CONSUMPTION_START_DATE']
    consumption_yearly['days'] = consumption_yearly['days'].dt.days
    consumption_yearly['CONSUMPTION'] = consumption_yearly['days'] * consumption_yearly['consumption_per_day']
    consumption_yearly['year'] = consumption_yearly['CONSUMPTION_START_DATE'].dt.year
    consumption_yearly.drop(consumption_yearly[consumption_yearly['CONSUMPTION_START_DATE'].dt.year < 2010].index,inplace=True)
    consumption_yearly_grouped = consumption_yearly.groupby(['CONSUMPTION_OBJECT_ID','year']).sum().reset_index()
    consumption_yearly_grouped = consumption_yearly_grouped.merge(df_categories, on='CONSUMPTION_OBJECT_ID',how='left')

    return consumption_yearly_grouped












