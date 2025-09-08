from ast import Dict
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
from surprise import Dataset, Reader, KNNBasic


#class with the functions that train the models evaluate his preformence and improve him
class TrainModels:
    
        """
        create prediction for recommendation by using
        random forest regression it get splited data from PreProcessingData as tuple
        and save the trained model
        """
        @staticmethod
        def context_base_RandomForestRegressor_train_model(x_train,y_train):
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            print("training the model")
            model.fit(x_train,y_train)
            print("training complete")
            joblib.dump(model,"context_based_movie_model")
            return model
        
        
        #evaluate the context based model
        #get the model from context_base_train_model and x test y test
        @staticmethod
        def evaluate_context_based_model(model: BaseEstimator ,x_test:pd.DataFrame ,y_test:pd.Series):
            prediction = model.predict(x_test)
            rmse = mean_squared_error(y_test, prediction, squared=False)
            print(f"the rmse of the model is: {rmse}")

        
        #context base model improve hyperparameters using randomizedSearchCV
        @staticmethod
        def context_model_hyperparameters_improvement(x_train: pd.DataFrame, y_train:pd.DataFrame):
            param_dict = {
                "n_estimators" : [100, 200, 400, 600, 800, 1000],
                "max_depth": [10, 20, 30, None],
                "min_samples_split": [2, 5, 10]
            }
            
            random_search = RandomizedSearchCV(
                estimator=RandomForestRegressor(random_state=42),
                param_distributions=param_dict,
                n_iter=10,
                cv=3,
                random_state=42,
                verbose=2,
                n_jobs=-1
            )
            print("start hyperparameters search")
            random_search.fit(x_train,y_train)
            print(f"\nBest parameters found: {random_search.best_params_}")
            joblib.dump(random_search.best_estimator_,"context_based_improve_model.pkl")
            return random_search.best_estimator_
        
        #building extra model to try and improve the RandomForestRegressor context based model
        #by using 
        @staticmethod
        def context_based_gradientBoosting_train_model(x_train:pd.DataFrame, y_train:pd.Series):
            model = xgb.XGBRegressor(n_estimators=100, random_state=42)
            print("start train the model")
            model.fit(x_train,y_train)
            print("training complete")
            joblib.dump(model,"context_based_gradientBoosting_model.pkl")
            return model
            
        
        #improve gradient boosting hyperparameters
        @staticmethod
        def gradient_boosting_hyperparameters_improvement(x_train:pd.DataFrame, y_train:pd.DataFrame):
            param_dict = {
                "n_estimators" : [100, 200, 400, 600, 800, 1000],
                "learning_rate" : [0.1,0.2,0.3,0.4,0.5],
                "max_depth" : [3,4,5,6,7,8],
                "subsample" : [0.2,0.4,0.6,0.8],
                "colsample_bytree":[0.6,0.7,0.8,0.9]
            }
            
            random_search = RandomizedSearchCV(
                estimator = xgb.XGBRegressor(random_state=42),
                param_distributions=param_dict,
                n_iter=10,
                cv=3,
                random_state=42,
                verbose=2,
                n_jobs=-1
            )
            print("start huperparameters search")
            random_search.fit(x_train,y_train)
            print(f"\nBest parameters found: {random_search.best_params_}")
            joblib.dump(random_search.best_estimator_,"gradient_boosting_improvement.pkl")
            return random_search.best_estimator_
        
        
        ## create user based model! ##
        
        
        
#class that use the trained models that we have and use them to give us
# a presiction for a recommended movie
class HybridRecommend:
    
    #get all the moovies the user havnt seen yet
    #get data_dict from DataCleaning data_pipeline 
    @staticmethod 
    def movies_to_recommend(user_id:int, clean_data: Dict[str,pd.DataFrame])->pd.DataFrame:
        movies_df = clean_data["movies.csv"]
        rating_df = clean_data["rating.csv"]
        
        seen_movies = rating_df[rating_df["userId"] == user_id]["movieId"].unique()
        unseen_movies = movies_df[~movies_df["movieId"].isin(seen_movies)]
        
        return unseen_movies

    #get context based data frame with the unseen moovies of the user
    #take that and make a prediction and return the top 5 outcome
    @staticmethod
    def context_recommend_random_fores_regression(df:pd.DataFrame, model:BaseEstimator)->pd.DataFrame:
        prediction = model.predict(df)
        df["prediction"] = prediction
        sorted_df = df.sort_values(by="prediction",ascending=False)
        print(sorted_df.head(5))
        return sorted_df.head(5)
         
    
    #train user based model using knn
    @staticmethod
    def user_based_model(df:pd.DataFrame):
        pass