import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd 
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransforation,DataTransformationConfig
from src.components.model_trainer import ModelTrainer,ModelTrainingConfig


@dataclass
class DataIngestionConfig:
    train_data_path : str = os.path.join('artifacts',"train.csv")
    test_data_path:str = os.path.join('artifacts',"test.csv")
    raw_data_path: str = os.path.join('artifacts',"raw_data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    def intiate_data_cofig(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df=pd.read_csv("notepad\stud.csv")
            logging.info("Read the Dataset")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info("Train and Test Data Intiated ")
            train_set,test_set=train_test_split(df,test_size=0.25,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("Ingestion Completed")
            return (
                self.ingestion_config.test_data_path,
                self.ingestion_config.train_data_path
                )
        except Exception as e:
            raise CustomException(sys,e)
    


if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.intiate_data_cofig()


    data_transformation=DataTransforation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

    model_trainer=ModelTrainer()
    print(model_trainer.initiate_model_train(train_arr=train_arr,test_arr=test_arr))







        