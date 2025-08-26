from typing import Dict
import pandas as pd 
import numpy as np
import matplotlib as plt
import os
from sklearn.preprocessing import MinMaxScaler
from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.preprocessing import StandardScaler

#class that dealing with importing the datasets
class ImportData:
    
    #import the data from kaggles 
    #return dictionary of dataFrames of the files 
    @staticmethod
    def import_data():
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files("garymk/movielens-25m-dataset",path='.',unzip=True)
        datafram_dict = {}
        file_path = "ml-25m"
        for file in os.listdir(file_path):
            if file.endswith(".csv"):
                df = pd.read_csv(os.path.join(file_path,file))
                datafram_dict[file_path] = df
        return datafram_dict
    
#class that handel the data cleaning
#get a dictionary of dataFrame and clean the data
#all the functions get dictionary of dataframes with the files from import_data
class DataCleaning:

    #check for missing data
    @staticmethod
    def data_check(dataset_dict:dict):
        cleaned_dataframes = {}
        for data in dataset_dict:
            new_data = data.dropna()
            cleaned_dataframes[data] = new_data
        return cleaned_dataframes
    
    #remove duplicates
    @staticmethod
    def remove_dup(dataset_list:dict):
        new_data = {}
        for file in dataset_list:
            undup_data = file.drop_duplicates()
            new_data[file] = undup_data
        return new_data
    
    
    
    #fitting the data types
    @staticmethod
    def data_fit(dataset_dict:dict):
        new_data = {}
        type_map ={ "ratings.csv":{"userId":"int64", "movieId":"int64", "rating":"float64", "timestamp":"int64"},
                    "tags.csv": {"userId":"int64", "movieId":"int64", "tag":"object", "timestamp":"int64"},
                    "movies.csv": {"movieId":"int64", "title":"object", "genres":"object"},
                    "links.csv" :{"movieId":"int64", "imdbId":"int64", "tmdbId":"int64"},
                    "genome-scores.csv": {"movieId":"int64", "tagId":"int64", "relevance":"float64"},
                    "genome-tags.csv": {"tagId":"int64", "tag":"object"}}
        
        for file_name, data in dataset_dict.items():
            if file_name in type_map:
                cur_type = type_map[file_name]
                new_data[file_name] = data.astype(cur_type)
            else:
                new_data[file_name] = data
                
        return new_data
    
    
    #normalize the data
    @staticmethod
    def data_normalaize(dataset_dict: dict):
        scaler = MinMaxScaler()
        
        for file_name in dataset_dict:
            df = dataset_dict[file_name]
            for data in df.columns:
                if pd.api.types.is_numeric_dtype(df[data]):
                    df["normalize_"+data] = scaler.fit_transform(df[[data]])
                    
        return dataset_dict
    
    
    #categorical the data
    #go through the dataframes and seperate the text cols for each element 
    #give 1 if the element within the col 0 otherwize
    @staticmethod
    def categorical_data(data_dict:dict):
        new_data_dict = {}
        for file_name, df in data_dict.items():
            text_cols = df.select_dtypes(include=["object"]).columns
            if not text_cols.empty:
                dummies = pd.get_dummies(df[text_cols])
                df = df.drop(columns=text_cols)
                df = pd.concat([df,dummies],axis=1)
            new_data_dict[file_name] = df
            
        return new_data_dict
    
    #scaling the data. standartizayion
    @staticmethod
    def standart_data(data_dict:dict):
        scalered_dict = {}
        for file_name, df in data_dict.items():
            df_copy = df.copy()
            scaler = StandardScaler()
            scaled_data_arr = scaler.fit_transform(df_copy)
            scaled_df = pd.DataFrame(scaled_data_arr, columns = df_copy.columns)
            scalered_dict[file_name] = scaled_df
        
        return scalered_dict
    
    
    #convert all timestamp columns to datatime
    @staticmethod
    def to_time_date(data_dict:Dict[str,pd.DataFrame]):
        for file_name, df in data_dict.items():
            for col in df.columns:
                if "timestamp" in col.lower() or "date" in col.lower():
                    if pd.api.types.is_numeric_dtype(df[col]):
                        df[col] = pd.to_datetime(df[col], unit='s',errors="coerce") #unit='s' treat the number as secound count
                    elif pd.api.types.is_object_dtype(df[col]):
                        df[col] = pd.to_datetime(df[col],errors="coerce")
        
        return data_dict
            
    
    #class that handel aggregation
    #create aggregation functions for tha data
    class aggregation:
        
        """
        user aggregation
        """
        
        #show for each user how much movies he rated
        @staticmethod
        def user_rating_amount(data_dict:dict):
            rating_df = data_dict["ratings.csv"]
            user_rating = rating_df["userId"].values_counts()  
            return user_rating
        
        #average rating per user
        @staticmethod
        def mean_user_rating(data_dict:Dict[str, pd.DataFrame]):
            rating_df = data_dict["ratings.csv"]
            user_avg_rating = rating_df.groupby("userId")["rating"].mean()
            return user_avg_rating
        
        #get the standart deviation of each user rating
        @staticmethod
        def user_std(data_dict:Dict[str, pd.DataFrame]):
            std_df = data_dict["ratings.csv"]
            user_std_df = std_df.groupby("userId")["rating"].std()
            return user_std_df
        
        #show the average frequency of a user in the system
        @staticmethod
        def user_avg_freq(data_dict:Dict[str, pd.DataFrame]):
            date_time_df = DataCleaning.to_time_date(data_dict)
            df = date_time_df["rating.csv"]
            sorted_df = df.sort_values(by=["userId","timestamp"])
            sorted_df["event_time_diff"] = sorted_df.groupby("userId")["timestamp"].diff()
            user_avg_freq = sorted_df.groupby("userId")["event_time_diff"].mean()
            return user_avg_freq
            
        
        """
        movie aggregation
        """
        
        #show the mean rating for each movie
        @staticmethod
        def mean_movie_rate(data_dict: Dict[str,pd.DataFrame]):
            rating_df = data_dict["ratings.csv"]
            movie_rate = rating_df.groupby("movieId")["rating"].mean()
            return movie_rate
        
        #overall number of rating per movie
        @staticmethod
        def rating_amount_per_movie(data_dict:Dict[str,pd.DataFrame]):
            rating_df = data_dict["ratings.csv"]
            rating_amount = rating_df["movieId"].value_counts()
            return rating_amount
        
        
        #find the standart deviation of each movie rating
        @staticmethod
        def movie_std(data_dict:Dict[str,pd.DataFrame]):
            std_df = data_dict["ratings.csv"]
            movie_std_df = std_df.groupby("movieId")["rating"].std()
            return movie_std_df
        
        
        
        