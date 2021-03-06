from process_output_data import process_output_data
from process_airbnb_data import process_airbnb_data
from helper_functions.pickling import load_obj
import pandas as pd
import logging
import pickle
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.NOTSET)

# Output_per_object_regional
logging.info("Loading Output data.")
postcode_dict = load_obj('../pipeline_data/postaldict')
output_data = pd.read_csv("../pipeline_data/Export_Verbruik_2010-2019_anon.csv") #("../pipeline_data/100_test.csv") # 
processed_data = Path("../pipeline_data/clean_output_data.csv")


if processed_data.is_file():
    logging.info("Check for new consumption IDs")
    processed_ids = pd.read_csv("../pipeline_data/processed_ids.csv")
    processed_ids_array = processed_ids.processed_id.unique()
    output_ids = output_data.VERBRUIK_ID.unique()
    new_ids = np.setdiff1d(output_ids,processed_ids_array)

    if len(new_ids) > 0:
        logging.info("Processing new Output data.{} rows".format(len(new_ids)))
        new_output_data = output_data[output_data['VERBRUIK_ID'].isin(new_ids)]
        cleaned_new_output_data = process_output_data(new_output_data,postcode_dict)
        clean_output_data = pd.concat([processed_data, cleaned_new_output_data], ignore_index=True)
        clean_output_data.to_csv("../pipeline_data/clean_output_data.csv",index=False)
    else:
        logging.info("No new Output data.")

else:
    logging.info("Processing Output data.")
    clean_output_data = process_output_data(output_data,postcode_dict)
    clean_output_data.to_csv("../pipeline_data/clean_output_data.csv",index=False)


logging.info("Output data processed.")
clean_output_data = pd.read_csv("../pipeline_data/clean_output_data.csv")
print(clean_output_data.head(), clean_output_data.shape)


# AirBnB
logging.info("Loading AirBnB data.")
reviews = pd.read_csv("../pipeline_data/airBnB_reviews.csv")
listings = pd.read_csv("../pipeline_data/airBnB_listings.csv")
region_mapping = load_obj('../pipeline_data/region_map_dic')
#ratio_data

logging.info("Processing AirBnB data.")
airbnb_data = process_airbnb_data(reviews,listings,region_mapping)
logging.info("AirBnB data processed.")
# save airbnb data
print(airbnb_data.head())



# join data