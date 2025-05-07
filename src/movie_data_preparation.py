import pandas as pd
import os
import requests
from io import BytesIO
from zipfile import ZipFile


def download_and_prepare_movielens():
    """
    Downloads the MovieLens Small dataset and prepares it for use
    """
    print("Downloading MovieLens dataset...")
    
    # Create data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # Download the dataset
    url = 'https://files.grouplens.org/datasets/movielens/ml-latest.zip'
    response = requests.get(url)
    
    # Extract the dataset
    with ZipFile(BytesIO(response.content)) as zip_file:
        zip_file.extractall('data')
    
    # Load the movies and ratings data
    movies_df = pd.read_csv('data/ml-latest/movies.csv')
    ratings_df = pd.read_csv('data/ml-latest/ratings.csv')
    
    # Process the data
    # Extract year from title and create a clean title column
    movies_df['year'] = movies_df['title'].str.extract(r'\((\d{4})\)$')
    movies_df['clean_title'] = movies_df['title'].str.replace(r'\s*\(\d{4}\)$', '', regex=True)
    
    # Split genres into a list
    movies_df['genres'] = movies_df['genres'].str.split('|')
    
    # Calculate average rating for each movie
    avg_ratings = ratings_df.groupby('movieId')['rating'].mean().reset_index()
    avg_ratings.columns = ['movieId', 'avg_rating']
    
    # Count the number of ratings for each movie
    rating_counts = ratings_df.groupby('movieId')['rating'].count().reset_index()
    rating_counts.columns = ['movieId', 'rating_count']
    
    # Merge the average ratings and rating counts with the movies data
    movies_df = pd.merge(movies_df, avg_ratings, on='movieId', how='left')
    movies_df = pd.merge(movies_df, rating_counts, on='movieId', how='left')
    
    # Fill NaN values
    movies_df['avg_rating'] = movies_df['avg_rating'].fillna(0)
    movies_df['rating_count'] = movies_df['rating_count'].fillna(0)
    
    # Save the processed data
    movies_df.to_csv('data/processed_movies.csv', index=False)
    
    print(f"Data prepared successfully! Total movies: {len(movies_df)}")
    return movies_df

if __name__ == "__main__":
    movies_df = download_and_prepare_movielens()
    print(movies_df.head())