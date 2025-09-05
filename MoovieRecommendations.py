from typing import Tuple
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error

#class with the functions that train the models evaluate his preformence and improve him
class TrainModels:
    
        """
        create prediction for recommendation by using
        random forest regression it get splited data from PreProcessingData as tuple
        """
        @staticmethod
        def context_base_train_model(x_train,y_train):
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            print("training the model")
            model.fit(x_train,y_train)
            print("training complete")
            return model
        
        
        #evaluate the context based model
        #get the model from context_base_train_model and x test y test
        @staticmethod
        def evaluate_context_based_model(model: BaseEstimator ,x_test:pd.DataFrame ,y_test:pd.Series):
            prediction = model.predict(x_test)
            rmse = mean_squared_error(y_test, prediction, squared=False)
            print(f"the rmse of the model is: {rmse}")

        
        
class HybridRecommend:
    
    pass