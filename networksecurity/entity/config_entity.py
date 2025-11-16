from datetime import datetime
import os 
from networksecurity.constant import training_pipeline

print(training_pipeline.PIPELINE_NAME)
print(training_pipeline.ARTIFACT_DIR)

class TrainingPipelineConfig:
    def __init__(self,timestamp=datetime.now()):                                ### class that holds global config for ml pipeline
        timestamp=timestamp.strftime("%m_%d_%Y_%H_%M_%S")
        self.pipeline_name=training_pipeline.PIPELINE_NAME                      ## read pipeline name and artifact base folder from constant 
        self.artifact_name=training_pipeline.ARTIFACT_DIR                       ## read pipeline name and artifact base folder from constant
        self.artifact_dir=os.path.join(self.artifact_name,timestamp)            
        self.model_dir=os.path.join("final_model")                              ##path where model will be saved 
        self.timestamp:str=timestamp

class DataIngestionConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):         ## Uses the main pipeline config to build ingestion-specific directories.
        self.data_ingestion_dir:str=os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.DATA_INGESTION_DIR_NAME                           ## This tells your pipeline where to store: downloaded raw data  .ingested data .logs related to ingestion .example:artifacts/data_ingestion

        )

        self.feature_store_file_path:str=os.path.join(
            self.data_ingestion_dir,                                            ## This is the base directory created earlier
            training_pipeline.DATA_INGESTION_FEATURE_STORE_DIR,                 ## This is a directory name defined as a constant in your training_pipeline module. "feature_store"  path becomes artifacts/data_ingestion/feature_store  (stores cleaned data)
            training_pipeline.FILE_NAME                                         ## "phising.csv"  artifacts/data_ingestion/feature_store/phising.csv

            )
        
        self.training_file_path:str=os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_INGESTED_DIR,                      
            training_pipeline.TRAIN_FILE_NAME                                   ## train.csv file 
            
            ##  Store the training portion of the ingested dataset inside the ingestion directory, under an 'ingested' subfolder   
            )
        
        self.testing_file_path: str = os.path.join(
                self.data_ingestion_dir, 
                training_pipeline.DATA_INGESTION_INGESTED_DIR, 
                training_pipeline.TEST_FILE_NAME
            )
        
        self.train_test_split_ratio:float=training_pipeline.DATA_INGESTION_TRAIN_TEST_SPLIT_RATION     ## This loads a constant value that defines what ratio to use for splitting dataset into train/test.
        self.collection_name:str =training_pipeline.DATA_INGESTION_COLLECTION_NAME                     ## Loads the MongoDB (or any DB) collection name where raw data is stored.
        self.database_name: str =training_pipeline.DATA_INGESTION_DATABASE_NAME                        ## Loads the database name containing the collection
                                


class DataValidationConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.data_validation_dir: str = os.path.join( training_pipeline_config.artifact_dir, 
                                                     training_pipeline.DATA_VALIDATION_DIR_NAME)
        
        self.valid_data_dir: str = os.path.join(self.data_validation_dir, 
                                                training_pipeline.DATA_VALIDATION_VALID_DIR)
        
        self.invalid_data_dir: str = os.path.join(self.data_validation_dir, 
                                                  training_pipeline.DATA_VALIDATION_INVALID_DIR)
        
        self.valid_train_file_path: str = os.path.join(self.valid_data_dir, 
                                                       training_pipeline.TRAIN_FILE_NAME)
        
        self.valid_test_file_path: str = os.path.join(self.valid_data_dir, 
                                                      training_pipeline.TEST_FILE_NAME)
        self.invalid_train_file_path: str = os.path.join(self.invalid_data_dir, 
                                                         training_pipeline.TRAIN_FILE_NAME)
        self.invalid_test_file_path: str = os.path.join(self.invalid_data_dir, 
                                                        training_pipeline.TEST_FILE_NAME)
        self.drift_report_file_path: str = os.path.join(
            self.data_validation_dir,
            training_pipeline.DATA_VALIDATION_DRIFT_REPORT_DIR,
            training_pipeline.DATA_VALIDATION_DRIFT_REPORT_FILE_NAME,

        
        )
        '''
               The DataValidationConfig class creates all the folder paths needed for validating the dataset. 
               It separates valid and invalid data, stores cleaned train/test files, and creates a place for the data drift report. 
               Its purpose is to organize everything the validation step needs so the pipeline runs smoothly.
               
        '''

class DataTransformationConfig:
     def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.data_transformation_dir: str = os.path.join( training_pipeline_config.artifact_dir,training_pipeline.DATA_TRANSFORMATION_DIR_NAME )
        self.transformed_train_file_path: str = os.path.join( self.data_transformation_dir,training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            training_pipeline.TRAIN_FILE_NAME.replace("csv", "npy"),)
        self.transformed_test_file_path: str = os.path.join(self.data_transformation_dir,  training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            training_pipeline.TEST_FILE_NAME.replace("csv", "npy"), )
        self.transformed_object_file_path: str = os.path.join( self.data_transformation_dir, training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
            training_pipeline.PREPROCESSING_OBJECT_FILE_NAME,)
        

        '''
             The DataTransformationConfig class creates all folder paths needed for storing transformed training data, transformed test data, and the saved preprocessing object. 
             It converts CSV names into .npy format and organizes everything inside the data_transformation directory so the pipeline can easily find its transformed datasets and preprocessing pipeline.
        '''
class ModelTrainerConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.model_trainer_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, training_pipeline.MODEL_TRAINER_DIR_NAME
        )
        self.trained_model_file_path: str = os.path.join(
            self.model_trainer_dir, training_pipeline.MODEL_TRAINER_TRAINED_MODEL_DIR, 
            training_pipeline.MODEL_FILE_NAME
        )
        self.expected_accuracy: float = training_pipeline.MODEL_TRAINER_EXPECTED_SCORE
        self.overfitting_underfitting_threshold = training_pipeline.MODEL_TRAINER_OVER_FIITING_UNDER_FITTING_THRESHOLD

        '''
             The ModelTrainerConfig class organizes where the trained ML model will be saved and sets important training rules such as required accuracy and the allowed overfitting/underfitting limit. 
             It ensures the training step knows where to save results and how to judge if the model is good enough.  
        
        '''