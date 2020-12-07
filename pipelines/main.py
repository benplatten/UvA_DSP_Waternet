from process_airbnb_data import process_airbnb_data
from helper_functions.pickling import load_obj
import pandas as pd
import logging
import pickle



# AirBnB
logging.info("Loading AirBnB data.")
reviews = pd.read_csv("data/airBnB_reviews.csv")
listings = pd.read_csv("data/airBnB_listings.csv")
region_mapping = load_obj('data/region_map_dic')
#ratio_data

logging.info("Processing AirBnB data.")
airbnb_data = process_airbnb_data(reviews,listings,region_mapping)
logging.info("AirBnB data processed.")
print(airbnb_data.head())