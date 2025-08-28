from typing import Dict
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt            
            
        
#create class for visualization
#contain matplotlib based functions
class visualization:
        
        """
        Distribution and rankings visualization
        """
        
        #create histogram of rating distribution
        def rate_diversity(data_dict:Dict[str,pd.DataFrame]):
            df = data_dict["ratings.csv"]
            df["rating"].hist(bins=[0.5,1.5,2.5,3.5,4.5,5.5],edgecolor='black')
            plt.title("rating distribution")
            plt.xlabel("rating")
            plt.ylabel("numbers of rating")
            plt.show()            

        #show the mean rating for genre
        #get series from genre_avg_rating
        def avg_genre_rate(series:pd.Series):
            series.plot(kind="bar")
            plt.title("average genre rating")
            plt.xlabel("genre")
            plt.ylabel("avg rate")
            plt.xticks(rotation=45,ha="right")
            plt.show()
            
        
        #show the distribution of user avg rating
        #get series from mean_user_rating
        @staticmethod
        def user_avg_rating(series:pd.Series):
            series.plot(kind="hist",bins=30,edgecolor="black")
            plt.title("avg user rating")
            plt.xlabel("Average Rating")
            plt.ylabel("Number of Users")
            plt.show()
            
        
        #show the average rate per movie
        #get series from mean_movie_rate
        @staticmethod
        def movie_avg_rate(series:pd.Series):
            series.plot(kind="hist",bins=30,edgecolor="black")
            plt.title("Distribution of Average Movie Ratings")
            plt.ylabel("number of movies")
            plt.xlabel("avg rating")
            plt.show()
            

        """
        Popularity and engagement
        """
        
        #show the most 20 popular movies
        #get series of top 20 movies and theire rate from top_20_movies from aggregation class
        @staticmethod
        def top_20_movies(series:pd.Series):
            series.plot(kind="bar")
            plt.title("top 20 rating movies")
            plt.xlabel("movies")
            plt.ylabel("rate")
            plt.xticks(rotation=45,ha="right")
            plt.show()
            
