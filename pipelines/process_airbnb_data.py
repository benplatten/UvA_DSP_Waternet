import pandas as pd
import numpy as np
import logging
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.NOTSET)

def process_airbnb_data(reviews,listings,region_mapping): # ratio_data):

    reviews = reviews # should be passed to function as a dataframe
    listings = listings
    region_mapping = region_mapping # pass as dictionary. save somewhere as dictionary

    reviews['date'] = reviews['date'].apply(lambda x: pd.to_datetime(x, format='%Y-%m-%d'))
    reviews = reviews[reviews['date'] >= "2010-01-01"]

    listings_smol = listings[['id','neighbourhood','latitude', 'longitude']]
    listings_smol.rename(columns={"id": "listing_id"},inplace=True)
    listings_smol['region'] = listings_smol['neighbourhood'].map(region_mapping)

    unmatched_regions = listings_smol['neighbourhood'][listings_smol['region'].isna()].to_list() 
    if len(unmatched_regions) > 0:
        logging.info("No mapping region: {}".format(' '.join(map(str, unmatched_regions))))
    else:
        logging.info("All regions mapped")
    

    listings_smol.drop(['neighbourhood'],inplace=True,axis=1)

    daily_reviews_borough = reviews.merge(listings_smol,on='listing_id', how='left')

    unmatched_listings = daily_reviews_borough['listing_id'][daily_reviews_borough.isna().any(axis=1)].to_list()
    if len(unmatched_listings) > 0:
        logging.info("No match for listing: {}".format(' '.join(map(str, unmatched_listings))))
    else:
        logging.info("All listings matched")

    daily_reviews_borough.dropna(inplace=True)
    daily_reviews_borough =  daily_reviews_borough.groupby(['date','region']).count().reset_index()
    daily_reviews_borough['reviews'] = daily_reviews_borough['listing_id']
    daily_reviews_borough.drop(['listing_id','latitude','longitude'],inplace=True,axis=1)
    daily_reviews_borough.sort_values('date')
    daily_reviews_borough = pd.pivot_table(daily_reviews_borough, values='reviews', index=['date'],
                    columns=['region'], aggfunc=np.sum,fill_value=0)

    return daily_reviews_borough
    













