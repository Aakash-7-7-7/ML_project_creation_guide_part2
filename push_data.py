import os
import sys
import json
import certifi
import pandas as pd 
import numpy as np 
import pymongo
from networksecurity.exception.exception import CustomException
from networksecurity.logging.logger import logging

from dotenv import load_dotenv
load_dotenv()

MONGO_DB_url=os.getenv("MONGO_DB_url") ## Reads your MongoDB connection string from the environment file.

class NetworkDataExtract():
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise CustomException(e,sys)
    def csv_to_json_convertor(self,file_path):
        try:
            data=pd.read_csv(file_path)
            data.reset_index(drop=True,inplace=True)##Removes the old index. Ensures continuous 0,1,2,... index. Prevents index from becoming part of the final JSON.
            records=list(json.loads(data.T.to_json()).values())
            return records
        except Exception as e:
            raise CustomException(e,sys)

    def insert_data_mongodb(self,records,database,collection):
        try:
            self.database=database
            self.collection=collection
            self.records=records

            self.mongo_client=pymongo.MongoClient(MONGO_DB_url)
            self.database=self.mongo_client[self.database] #Selects or creates the MongoDB database.
            
            self.collection=self.database[self.collection]#Selects or creates the collection inside the database.
            self.collection.insert_many(self.records)
            return(len(self.records))
        except Exception as e:
            raise CustomException(e,sys)

if __name__ == '__main__':
    try:
        FILE_PATH = r'Network_Data\phisingData.csv'
        DATABASE = 'AAKASH'
        COLLECTION = 'NetworkData'

        logging.info("CSV to JSON Conversion started")
        networkobj = NetworkDataExtract()
        records = networkobj.csv_to_json_convertor(FILE_PATH)
        logging.info("CSV to JSON Conversion completed")

        logging.info("MongoDB insertion started")
        count = networkobj.insert_data_mongodb(records, DATABASE, COLLECTION)
        logging.info(f"MongoDB insertion completed. Total records inserted: {count}")

    except Exception as e:
        raise CustomException(e, sys)