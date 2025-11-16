from networksecurity.exception.exception import CustomException
from networksecurity.logging.logger import logging
import sys

from networksecurity.components.data_injestion import DataIngestion
from networksecurity.entity.config_entity import DataIngestionConfig
from networksecurity.entity.config_entity import TrainingPipelineConfig

if __name__=='__main__':
    try:
        trainingpipeline=TrainingPipelineConfig()
        dataingestionconfig=DataIngestionConfig(trainingpipeline)
        data_injestion=DataIngestion(dataingestionconfig)
        dataingestionartifact=data_injestion.initiate_data_ingestion()
        print(dataingestionartifact)

        logging.info("initiate the data ingestion")
        

    except Exception as e:
        raise CustomException(e,sys)