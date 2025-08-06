import pandas as pd 
import numpy as np
import matplotlib as plt
import os
from kaggle.api.kaggle_api_extended import KaggleApi

#class that dealing with importing the datasets
class ImportData:
    
    #import the data from kaggles 
    #return list of dataFrames of the files 
    @staticmethod
    def import_data():
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files("garymk/movielens-25m-dataset",path='.',unzip=True)
        datafram_list = []
        file_path = "ml-25m"
        for file in os.listdir(file_path):
            if file.endswith(".csv"):
                df = pd.read_csv(os.path.join(file_path,file))
                datafram_list.append(df)
        return datafram_list
    
#class that handel the data cleaning
#get a list of dataFrame and clean the data
#all the functions get the dataframe with the files from import_data
class DataCleaning:

    #check for missing data
    @staticmethod
    def data_check(dataset_list):
        cleaned_dataframes = []
        for data in dataset_list:
            new_data = data.dropna()
            cleaned_dataframes.append(new_data)
        return cleaned_dataframes
    
    #dealing with duplicates
    @staticmethod
    def remove_dup(dataset_list):
        new_data = []
        for file in dataset_list:
            undup_data = file.drop_duplicates()
            new_data.append(undup_data)
        return new_data
    
    
    
    
    
    #fitting the data types
    @staticmethod
    def data_fit(dataset_list):
        new_data = []
        rating_dict = {"userId":"int64", "movieId":"int64", "rating":"float64", "timestamp":"int64"}
        tag_dict = {"userId":"int64", "movieId":"int64", "tag":"object", "timestamp":"int64"}
        moovies_dict = {"movieId":"int64", "title":"object", "genres":"object"}
        link_dict = {"movieId":"int64", "imdbId":"int64", "tmdbId":"int64"}
        genome_score_dict = {"movieId":"int64", "tagId":"int64", "relevance":"float64"}
        genome_tag_dict = {"tagId":"int64", "tag":"object"}
        
        for data in dataset_list:
                column_header = data.columns
                col_list = list(column_header)            
                if col_list == ["tagId", "tag"]:
                    new_genome_tags_data = data.astype(genome_tag_dict)
                    new_data.append(new_genome_tags_data)
                elif col_list == ["movieId", "tagId", "relevance"]:
                    new_genome_scores_data = data.astype(genome_score_dict)
                    new_data.append(new_genome_scores_data)
                elif col_list == ["movieId", "imdbId", "tmdbId"]:
                    new_links_data = data.astype(link_dict)
                    new_data.append(new_links_data)
                elif col_list == ["movieId", "title", "genres"]:
                    new_movies_data = data.astype(moovies_dict)
                    new_data.append(new_movies_data)
                elif col_list == ["userId", "movieId", "tag", "timestamp"]:
                    new_tags_data = data.astype(tag_dict)
                    new_data.append(new_tags_data)
                elif col_list == ["userId", "movieId", "rating", "timestamp"]:
                    new_ratings_data = data.astype(rating_dict)
                    new_data.append(new_ratings_data)
                
        return new_data
    
    
    