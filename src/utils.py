import os
import sys
from src.exception import CustomException
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
import pickle





def save_object(file_path,obj):
    '''
    This function will make the file path and dump the data in pickle format.
    '''
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    


def evaluate_model(X_train,y_train,X_test,y_test,models,param):
    '''
    This Function will train the model and provides the r2_square report 
    '''
    report={}
    try:
        for  model_name,model in models.items():

            para=param[model_name]   #example here para={'n_estimators': [8,16,32,64,128,256]},if model_name=Random forest
            #model.fit(X_train,y_train)
            gs=GridSearchCV(model,para,cv=3)    #Running Grid Search for best paramters 
            gs.fit(X_train,y_train)    


            model.set_params(**gs.best_params_)   # Setting the new and best parameters for model
            model.fit(X_train,y_train)     #Training the model with new parameters 
            y_train_predict=model.predict(X_train)
            y_test_predict=model.predict(X_test)
            train_model_score=r2_score(y_train,y_train_predict)
            test_model_score=r2_score(y_test,y_test_predict)
            report[model_name]=test_model_score

            return report
    except Exception as e:
        raise CustomException(e,sys)
    


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)

        







