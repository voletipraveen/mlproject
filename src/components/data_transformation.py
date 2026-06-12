
import os
import sys
import pandas as pd
import numpy as np

#Sk-learn Models
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object



@dataclass
class DataTransformationConfig:
    preprocessor_obj_file:str=os.path.join('artifacts',"preprocessor.pkl")

class DataTransforation:
    def __init__(self):
        self.data_tranformation_config=DataTransformationConfig()

    def get_transformer_obj(self):
        '''
        This fun will transform the data based on catgeories
        '''
        try:
            num_cols=["reading_score","writing_score"]
            cat_cols=["gender",
                      "race_ethnicity",
                      "parental_level_of_education",
                      "lunch",
                      "test_preparation_course"
                      ]
            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                    ]
            )
            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("encoded",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )
            logging.info(f"num_colunms:{num_cols}")
            logging.info(f"cat_columns:{cat_cols}")
            preprocessor=ColumnTransformer(
                [
                    ("num_pipeliner",num_pipeline,num_cols),
                    ("cat_pipeline",cat_pipeline,cat_cols)
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        



    def initiate_data_transformation(self,train_path,test_path):
        try:

            logging.info("Reading the test data and train data from the Artifacts folder")
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            preprocessing_obj=self.get_transformer_obj()
            logging.info("Obtained Preprocessor object")
            
            target_column="math_score"

            input_feature_train=train_df.drop(columns=[target_column],axis=1,)
            target_feature_train=train_df[target_column]

            input_feature_test=test_df.drop(columns=[target_column],axis=1)
            target_feature_test=test_df[target_column]

            logging.info("Applying the Prepoccessore steps to train and test data")
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test)
            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test)]

            
            save_object(
                file_path=self.data_tranformation_config.preprocessor_obj_file,
                obj=preprocessing_obj
            )

            logging.info("Saving the Preprocessing file")   


            return (
                train_arr,
                test_arr,
                self.data_tranformation_config.preprocessor_obj_file
            )
        except Exception as e:
            raise CustomException(e,sys)