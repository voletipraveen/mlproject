import os
import sys
from dataclasses import dataclass


from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.utils import save_object,evaluate_model
from src.logger import logging



@dataclass
class ModelTrainingConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainingConfig()
    
    def initiate_model_train(self,train_arr,test_arr,):
        try:
            logging.info("Spliting the test arr and train arr for model training")
            X_train=train_arr[:,:-1]
            y_train=train_arr[:,-1]
            X_test=test_arr[:,:-1]
            y_test=test_arr[:,-1]
            
            models={
                "Linear Regression":LinearRegression(),
                "KNN":KNeighborsRegressor(),
                "Decission Tree Regression":DecisionTreeRegressor(),
                "Ada Boost Regression":AdaBoostRegressor(),
                "Gradient Boost Regression":GradientBoostingRegressor(),
                "Cat Boosting Regression":CatBoostRegressor(),
                "Random Forest Regression":RandomForestRegressor(),
                "XG Boost Regression":XGBRegressor()
                }
            model_report:dict=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)

            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model=models[best_model_name]

            if best_model_score<0.60:
                raise CustomException("There is no best model in the above list of models... please tune the models")
            logging.info(f"The best model is {best_model}and r2 scocre is {best_model_score}")

            save_object(
                file_path=ModelTrainingConfig.trained_model_file_path,
                obj=best_model
            )

            predicted_test=best_model.predict(X_test)
            r2_square=r2_score(y_test,predicted_test)

            return r2_square
        except Exception as e:
            raise CustomException(e,sys)



