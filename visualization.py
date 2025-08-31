from typing import Dict
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt            
import seaborn as sns
from wordcloud import WordCloud
from sklearn.decomposition import PCA
import networkx as nx
        
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
            
        #show the top 20 active users
        #get series from top_20_users from aggregation class
        @staticmethod
        def top_20_users(series: pd.Series):
            series.plot(kind="bar")
            plt.title("top 20 actives users")
            plt.xlabel("suserId")
            plt.ylabel("rating amount")
            plt.show()
            
            
        #show the activity of the users by time
        #get series from user_activity in aggregation 
        @staticmethod
        def user_activity(series:pd.Series):
            series.plot(kind="line")
            plt.title("number of rating over time")
            plt.xlabel("date")
            plt.xticks(rotation=45,ha="right")
            plt.ylabel("number of rating")
            plt.show()
            
        # Heatmap of popular rating hours and days
        @staticmethod
        def plot_rating_heatmap(ratings_df: pd.DataFrame):
            df = ratings_df.copy()
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['dayofweek'] = df['timestamp'].dt.day_name()
            
            activity_pivot = df.groupby('dayofweek')['hour'].value_counts().unstack().fillna(0)
            activity_pivot = activity_pivot.reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
            
            plt.figure(figsize=(12, 7))
            sns.heatmap(activity_pivot, cmap="viridis", annot=True, fmt=".0f")
            plt.title("Rating Activity: Day of Week vs. Hour of Day")
            plt.xlabel("Hour of Day")
            plt.ylabel("Day of Week")
            plt.show()

        # Bar chart of the most popular genres
        @staticmethod
        def plot_popular_genres(movies_df: pd.DataFrame, top_n=15):
            genre_counts = movies_df['genres'].str.split('|').explode().value_counts()
            top_genres = genre_counts.head(top_n)
            
            top_genres.sort_values().plot(kind='barh', figsize=(10, 8), color='skyblue')
            plt.title(f'Top {top_n} Most Popular Genres')
            plt.xlabel('Number of Movies')
            plt.ylabel('Genre')
            plt.tight_layout()
            plt.show()

        # Scatter plot: average rating vs. popularity
        @staticmethod
        def plot_rating_vs_popularity(ratings_df: pd.DataFrame, min_ratings=100):
            avg_ratings = ratings_df.groupby('movieId')['rating'].mean()
            rating_counts = ratings_df.groupby('movieId')['rating'].count()
            movie_stats = pd.DataFrame({'avg_rating': avg_ratings, 'rating_count': rating_counts})
            
            popular_movies = movie_stats[movie_stats['rating_count'] >= min_ratings]
            
            sns.scatterplot(data=popular_movies, x='rating_count', y='avg_rating', alpha=0.5)
            plt.title('Popularity vs. Average Rating')
            plt.xlabel('Number of Ratings (Popularity)')
            plt.ylabel('Average Rating')
            plt.xscale('log')
            plt.show()

        # Heatmap of the correlation matrix
        @staticmethod
        def plot_correlation_heatmap(df: pd.DataFrame):
            numeric_df = df.select_dtypes(include=np.number)
            corr_matrix = numeric_df.corr()
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
            plt.title('Correlation Matrix of Numeric Features')
            plt.show()

        # Bubble chart: rating, popularity, and standard deviation
        @staticmethod
        def plot_bubble_chart(ratings_df: pd.DataFrame, min_ratings=100):
            movie_stats = ratings_df.groupby('movieId')['rating'].agg(['mean', 'count', 'std']).fillna(0)
            popular_movies = movie_stats[movie_stats['count'] >= min_ratings]
            
            plt.figure(figsize=(12, 8))
            sns.scatterplot(data=popular_movies, x='mean', y='count', size='std', hue='std',
                            sizes=(20, 1000), alpha=0.7, palette='viridis')
            plt.title('Movie Analysis: Rating vs. Popularity vs. Polarization')
            plt.xlabel('Average Rating (mean)')
            plt.ylabel('Number of Ratings (count)')
            plt.legend(title='Rating Std Dev (std)')
            plt.show()

        # Stacked bar chart of rating distribution by genre
        @staticmethod
        def plot_stacked_ratings_by_genre(combined_df: pd.DataFrame, top_n=10):
            genres_exploded = combined_df.copy()
            genres_exploded['genres'] = genres_exploded['genres'].str.split('|')
            genres_exploded = genres_exploded.explode('genres')
            
            genre_counts = genres_exploded['genres'].value_counts().head(top_n).index
            top_genres_df = genres_exploded[genres_exploded['genres'].isin(genre_counts)]
            
            rating_dist = top_genres_df.groupby(['genres', 'rating']).size().unstack().fillna(0)
            rating_dist_percent = rating_dist.apply(lambda x: x / x.sum(), axis=1)
            
            rating_dist_percent.plot(kind='barh', stacked=True, figsize=(12, 8), cmap='viridis')
            plt.title('Rating Distribution by Genre (Normalized)')
            plt.xlabel('Percentage of Ratings')
            plt.ylabel('Genre')
            plt.legend(title='Rating')
            plt.show()

        # Word cloud of the most common tags
        @staticmethod
        def plot_tags_wordcloud(tags_df: pd.DataFrame):
            tag_counts = tags_df['tag'].value_counts()
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(tag_counts)
            
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('Most Common User Tags')
            plt.show()

        # PCA plot for displaying movie similarity
        @staticmethod
        def plot_pca_similarity(ratings_df: pd.DataFrame, movies_df: pd.DataFrame, sample_users=5000):
            movie_user_matrix = ratings_df.pivot(index='movieId', columns='userId', values='rating').fillna(0)
            sampled_matrix = movie_user_matrix.iloc[:, :sample_users]
            
            pca = PCA(n_components=2)
            movie_pca = pca.fit_transform(sampled_matrix)
            
            plt.figure(figsize=(12, 8))
            plt.scatter(movie_pca[:, 0], movie_pca[:, 1], alpha=0.5)
            plt.title('2D PCA of Movie Similarity Based on User Ratings')
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.show()

        # Network graph of user-movie connections
        @staticmethod
        def plot_user_movie_network(ratings_df: pd.DataFrame, sample_size=100):
            sample_df = ratings_df.head(sample_size)
            
            G = nx.Graph()
            users = sample_df['userId'].unique()
            movies = sample_df['movieId'].unique()
            G.add_nodes_from(users, bipartite=0)
            G.add_nodes_from(movies, bipartite=1)
            
            G.add_edges_from(sample_df[['userId', 'movieId']].to_numpy())
            
            plt.figure(figsize=(12, 12))
            pos = nx.spring_layout(G, iterations=50)
            nx.draw(G, pos, with_labels=False, node_size=50, width=0.5)
            plt.title(f'Network Graph of Users and Movies (Sample of {sample_size} ratings)')
            plt.show()

        # Bar chart of model performance
        @staticmethod
        def plot_model_performance(performance_dict: Dict[str, float]):
            models = list(performance_dict.keys())
            scores = list(performance_dict.values())
            
            plt.figure(figsize=(8, 5))
            plt.bar(models, scores, color='salmon')
            plt.title('Recommendation Model Performance Comparison')
            plt.ylabel('RMSE or Precision Score')
            plt.ylim(min(scores) * 0.9, max(scores) * 1.05)
            plt.show()

        # Sankey diagram of user flow between genres
        @staticmethod
        def plot_genre_sankey(sankey_data: pd.DataFrame):
            all_nodes = list(pd.concat([sankey_data['source'], sankey_data['target']]).unique())
            label_map = {label: i for i, label in enumerate(all_nodes)}
            
            fig = go.Figure(data=[go.Sankey(
                node = dict(pad=15, thickness=20, label=all_nodes),
                link = dict(
                    source = sankey_data['source'].map(label_map),
                    target = sankey_data['target'].map(label_map),
                    value = sankey_data['value']
                ))])
            fig.update_layout(title_text="User Flow Between Genres", font_size=10)
            fig.show()

        # Treemap of the genre composition in the library
        @staticmethod
        def plot_genre_treemap(movies_df: pd.DataFrame):
            genre_counts = movies_df['genres'].str.split('|').explode().value_counts().reset_index()
            genre_counts.columns = ['genre', 'count']
            
            fig = px.treemap(genre_counts, path=[px.Constant("All Genres"), 'genre'], values='count')
            fig.update_layout(title_text='Genre Composition in the Movie Library', margin = dict(t=50, l=25, r=25, b=25))
            fig.show()