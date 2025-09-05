from typing import Dict,Tuple
import pandas as pd 
import os
from sklearn.preprocessing import MinMaxScaler
from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#class that dealing with importing the datasets
class ImportData:
    
    #import the data from kaggles 
    @staticmethod
    def import_data():
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files("garymk/movielens-25m-dataset",path='.',unzip=True)
        
    #return dictionary of dataFrames of the files 
    @staticmethod
    def to_dataframe():    
        datafram_dict = {}
        file_path = "ml-25m"
        for file in os.listdir(file_path):
            if file.endswith(".csv"):
                df = pd.read_csv(os.path.join(file_path,file))
                datafram_dict[file] = df
        return datafram_dict
    
#class that handel the data cleaning
#get a dictionary of dataFrame and clean the data
#all the functions get dictionary of dataframes with the files from import_data
class DataCleaning:

    #check for missing data
    @staticmethod
    def handle_missing_values(dataset_dict:dict):
        cleaned_dataframes = {}
        for file, df in dataset_dict.items():
            new_data = df.dropna()
            cleaned_dataframes[file] = new_data
        return cleaned_dataframes
    
    #remove duplicates
    @staticmethod
    def remove_dup(dataset_dict:dict):
        new_data = {}
        for file,df in dataset_dict.items():
            undup_data = df.drop_duplicates()
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
    
    
    #create a pipeline to process the data
    @staticmethod
    def data_pipeline(data_dict:Dict[str,pd.DataFrame]):
        data_dict = DataCleaning.data_fit(data_dict)
        data_dict = DataCleaning.remove_dup(data_dict)
        data_dict = DataCleaning.handle_missing_values(data_dict)
        data_dict = DataCleaning.to_time_date(data_dict)
        return data_dict
    
    
            
    
#class that handel aggregation
#create aggregation functions for tha data
class aggregation:
        
        """
        user aggregation
        """
        
        #show for each user how much movies he rated
        @staticmethod
        def user_rating_amount(data_dict:Dict[str,pd.DataFrame]):
            rating_df = data_dict["ratings.csv"]
            user_rating = rating_df["userId"].value_counts()  
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
            df = date_time_df["ratings.csv"]
            sorted_df = df.sort_values(by=["userId","timestamp"])
            sorted_df["event_time_diff"] = sorted_df.groupby("userId")["timestamp"].diff()
            user_avg_freq = sorted_df.groupby("userId")["event_time_diff"].mean()
            return user_avg_freq
            
        #show the top 20 actives users
        @staticmethod
        def top_20_users(data_dict:Dict[str, pd.DataFrame]):
            user_rate = aggregation.user_rating_amount(data_dict)
            sorted_series = user_rate.sort_values(ascending=False)
            return sorted_series.head(20)
        
        #get the activity of user by time
        @staticmethod
        def user_activity(data_dict:Dict[str,pd.DataFrame]):
            dict = DataCleaning.to_time_date(data_dict)
            df = dict["ratings.csv"]
            df_index = df.set_index("timestamp")
            monthly_activity = df_index["rating"].resample('M').size()
            return monthly_activity
        
        
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
        
        
        #the overall number if movie ratings to his popularity
        @staticmethod
        def movie_pop(data_dict:Dict[str,pd.DataFrame]):
            df = data_dict["rating.csv"]
            popularity_df = df.value_counts("movieId")
            return popularity_df
        
        #the number of tags per movie. show the complexity of the plot
        @staticmethod
        def tags_per_movie(data_dict:Dict[str,pd.DataFrame]):
            df = data_dict["tags.csv"]
            movie_tags = df.value_counts("movieId")
            return movie_tags
        
        #return the top 20 movies and their rate
        @staticmethod
        def top_20_movies(data_dict:Dict[str,pd.DataFrame]):
            combined_df = pd.merge(data_dict["ratings.csv"],data_dict["movies.csv"],on="movieId")
            movie_rate = combined_df.groupby("title")["rating"].mean()
            final_df = movie_rate.sort_values(ascending=False)
            return final_df.head(20)
            
        """
        genre aggregation
        """
        
        #average rating per genre
        @staticmethod
        def genre_avg_rating(data_dict:Dict[str,pd.DataFrame]):
            combined_data = pd.merge(data_dict["ratings.csv"],data_dict["movies.csv"],on="movieId")
            combined_data["genres"] = combined_data["genres"].str.split('|')
            explosed_df = combined_data.explode("genres")
            genre_avg_rating = explosed_df.groupby("genres")["rating"].mean()
            return genre_avg_rating
            
        
        #return the amount of moovies in genre
        @staticmethod
        def amount_in_genre(data_dict:Dict[str,pd.DataFrame]):
            df = data_dict["movies.csv"]
            split_df = df["genres"].str.split("|")
            explode_df = split_df.explode("genres")
            count_genres = explode_df["genres"].count_values()
            return count_genres
            
            
        #return the most polaraiz genre
        #its mean it return the genre with the highest dtandart deviation
        @staticmethod
        def most_polaraiz_genre(data_dict:Dict[str,pd.DataFrame]):
            combined_data = pd.merge(data_dict["ratings.csv"],data_dict["movies.csv"],on="movieId")
            combined_data["genres"] = combined_data["genres"].str.split("|")
            explode_df = combined_data.explode("genres")
            genre_polaraiz = explode_df.groupby("genres")["ratings"].std()
            most_polaraiz = genre_polaraiz.idxmax()
            return most_polaraiz
            
        
        #get the precentage of genre from all the moovies genres
        @staticmethod
        def genre_relative(data_dict:Dict[str,pd.DataFrame]):
            df = data_dict["movies.csv"]
            df["genres"] = df["genres"].str.split('|')
            explode_df = df.explode("genres")
            genres_relativity = explode_df["genres"].value_counts(normalize=True) * 100
            return genres_relativity
        
        
        
        """
        tags aggregation
        """
        
        #average rate per tag
        @staticmethod
        def tag_avg_rate(data_dict:Dict[str,pd.DataFrame]):
            combined_df = pd.merge(data_dict["ratings.csv"],data_dict["tags.csv"],on="movieId")
            avg_per_tag = combined_df.groupby("tag")["rating"].mean()
            return avg_per_tag
        
        #frequency of each tag
        @staticmethod
        def tag_freq(data_dict:Dict[str,pd.DataFrame]):
            df = data_dict["tags.csv"]
            tag_freq = df["tag"].value_counts()
            return tag_freq
        
        #user useg per tag
        @staticmethod
        def user_tag_use(data_dict:Dict[str,pd.DataFrame]):
            df = data_dict["tags.csv"]
            user_tags = df["userId"].value_counts()
            return user_tags
        
        #the best tag. the tag with the highest rate
        @staticmethod
        def best_tag(data_dict:Dict[str,pd.DataFrame]):
            combined_data = pd.merge(data_dict["ratings.csv"],data_dict["tags.csv"],on="movieId")
            mean_tag_rate = combined_data.groupby("tag")["rating"].mean()
            highest = mean_tag_rate.idxmax()
            return highest
        
        #get the most popular tag
        @staticmethod
        def most_popular_tag(data_dict:Dict[str,pd.DataFrame]):
            df = data_dict["tags.csv"]
            tags_count = df["tag"].value_counts()
            most_popular = tags_count.idxmax()
            return most_popular
            
        
#create a class to preprocess the data
class PreProcessData:
            
        #scaling the data. standartization
        @staticmethod
        def standart_data(data_dict:Dict[str,pd.DataFrame]):
            scalered_dict = {}
            for file_name, df in data_dict.items():
                df_copy = df.copy()
                scaler = StandardScaler()
                scaled_data_arr = scaler.fit_transform(df_copy)
                scaled_df = pd.DataFrame(scaled_data_arr, columns = df_copy.columns)
                scalered_dict[file_name] = scaled_df
            
            return scalered_dict
        
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
              
        #merge the dataframes: movies.csv, ratings.csv into one table for the model
        #and add column mean_movie_rate
        #the function get clean data dict from pipeline in DataClean class and the mean rating for each movie from mean_movie_rate in aggregation
        @staticmethod
        def merge_dataFrames(clean_data:Dict[str,pd.DataFrame],mean_rating:pd.Series):
            mean_merge = pd.merge(clean_data["ratings.csv"],mean_rating,on="movieId")
            merge_df = pd.merge(clean_data["movies.csv"],mean_merge,on="movieId")
            return merge_df
        
        #one hot encoded for genres seperate genres and
        #give 1 if the element within the col 0 otherwize
        @staticmethod
        def one_hot_encoding_genres(df : pd.DataFrame):
            dummies_df = df["genres"].str.get_dummies(sep='|')
            final_df = df.drop("genres",axis=1).join(dummies_df)
            return final_df
        
        #add mean movie rating in a column to the dataframe
        @staticmethod
        def add_mean_movie_rate(data_dict: Dict[str,pd.DataFrame]):
            ratings_df = data_dict["ratings.csv"]
            mean_series = ratings_df.groupby('movieId')['rating'].mean().rename("mean_movie_rate") 
            ratings_with_mean = pd.merge(ratings_df, mean_series, on='movieId', how='left')
            return ratings_with_mean
    
        
        #create data frame for context based recommender
        #it would contain ratings.csv, novies.csv, cols: mean_movie_rate, rating_count, year
        def context_based_dataFrame(clean_data: Dict[str,pd.DataFrame]) -> pd.DataFrame:
            movie_df = clean_data["movies.csv"].copy()
            rating_df = clean_data["ratings.csv"].copy()
            merge_df = pd.merge(movie_df,rating_df,on="movieId",how="left")
            merge_df["year"] = merge_df["title"].str.extract(r'\((\d{4})\)')  
            rate_count = merge_df.groupby("movieId")["rating"].transform("count")
            merge_df["rating_count"] = rate_count
            mean_rate = merge_df.groupby("movieId")["rating"].transform("mean")
            merge_df["mean_movie_rate"] = mean_rate
            return merge_df
        
        
        #divid the data to featur and target
        #get data frame from context_based_dataFrame
        #return tuple of the features dataFrame and the target vector
        @staticmethod
        def select_final_features(df:pd.DataFrame):
            target_vector = df["rating"]
            featur_data = df.drop(columns=["rating","movieId","title"], errors="ignore")
            return featur_data, target_vector
        
        
        #get a tuple of featur data and target vector from select_final_features
        #spliting the dasta into training set and testing set
        @staticmethod
        def split_data(data_tuple:Tuple[pd.DataFrame,pd.Series]):
            x,y = data_tuple
            x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=50)
            return x_train,x_test,y_train,y_test
        
        
        #create a pipeline to the data pre processing for context bas model
        @staticmethod
        def context_based_pipeline_data(clean_data:Dict[str,pd.DataFrame]):
            df = PreProcessData.context_based_dataFrame(clean_data)
            one_hot_encoded = PreProcessData.one_hot_encoding_genres(df)
            final_features = PreProcessData.select_final_features(one_hot_encoded)
            data_split = PreProcessData.split_data(final_features)
            return data_split