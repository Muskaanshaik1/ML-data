
import streamlit as st
import pandas as pd
import numpy as np
import os

# --- Configuration ---
DATA_DIR = 'ml-100k' # Directory where the MovieLens 100k data is extracted
MOVIES_FILE = os.path.join(DATA_DIR, 'u.item')
RATINGS_FILE = os.path.join(DATA_DIR, 'u.data')

# --- Helper Functions ---

@st.cache_data # Cache data loading to improve performance
def load_data():
    """
    Loads movie and rating data from the MovieLens 100k dataset.
    Processes movie genres and calculates average ratings.
    """
    try:
        # Load movies data (u.item)
        # Columns: movie id | movie title | release date | video release date |
        # IMDb URL | unknown | Action | Adventure | Animation |
        # Children's | Comedy | Crime | Documentary | Drama | Fantasy |
        # Film-Noir | Horror | Musical | Mystery | Romance | Sci-Fi |
        # Thriller | War | Western
        # The last 19 fields are binary (0/1) indicating genres.
        movies_df = pd.read_csv(
            MOVIES_FILE,
            sep='|',
            header=None,
            encoding='latin-1',
            names=[
                'movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url',
                'unknown', 'Action', 'Adventure', 'Animation', 'Children\'s',
                'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
                'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller',
                'War', 'Western'
            ],
            usecols=list(range(24)) # Use the first 24 columns
        )

        # Drop unnecessary columns for this recommender
        movies_df = movies_df.drop(columns=['release_date', 'video_release_date', 'imdb_url', 'unknown'])

        # Load ratings data (u.data)
        # Columns: user id | item id | rating | timestamp
        ratings_df = pd.read_csv(
            RATINGS_FILE,
            sep='\t',
            header=None,
            names=['user_id', 'movie_id', 'rating', 'timestamp']
        )

        # Calculate average rating for each movie
        average_ratings = ratings_df.groupby('movie_id')['rating'].mean().reset_index()
        average_ratings.rename(columns={'rating': 'average_rating'}, inplace=True)

        # Merge movies_df with average_ratings
        movies_df = pd.merge(movies_df, average_ratings, on='movie_id', how='left')

        # Fill NaN average_ratings with 0 or a sensible default for movies with no ratings
        movies_df['average_rating'] = movies_df['average_rating'].fillna(0)

        # Get the list of all unique genres
        genre_columns = [
            'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime',
            'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
            'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
        ]
        all_genres = []
        for col in genre_columns:
            # For each movie, if the genre column is 1, add the genre name to a list
            # This approach is for creating a comprehensive list of all possible genres
            # that can be selected by the user.
            if col in movies_df.columns:
                all_genres.append(col)

        return movies_df, all_genres

    except FileNotFoundError:
        st.error(f"Error: Dataset files not found. Please ensure '{DATA_DIR}' directory "
                 f"with 'u.item' and 'u.data' is in the same directory as 'app.py'. "
                 f"Refer to the GitHub guide for data download instructions.")
        st.stop() # Stop the app if files are not found
    except Exception as e:
        st.error(f"An error occurred while loading data: {e}")
        st.stop()

def get_recommendations(genre, movies_df, top_n=5):
    """
    Recommends top N movies for a given genre based on average ratings.

    Args:
        genre (str): The genre to recommend movies for.
        movies_df (pd.DataFrame): DataFrame containing movie data and average ratings.
        top_n (int): Number of top recommendations to return.

    Returns:
        list: A list of recommended movie titles.
    """
    if genre not in movies_df.columns:
        return [f"No movies found for genre: '{genre}'. Please select a valid genre."]

    # Filter movies that belong to the selected genre
    genre_movies = movies_df[movies_df[genre] == 1]

    if genre_movies.empty:
        return [f"No movies available for the genre '{genre}' in the dataset."]

    # Sort by average rating in descending order
    recommended_movies = genre_movies.sort_values(by='average_rating', ascending=False)

    # Return the top N movie titles
    return recommended_movies['title'].head(top_n).tolist()

# --- Streamlit UI ---
st.set_page_config(page_title="Genre-Based Movie Recommender", layout="centered")

st.title("🎬 Genre-Based Movie Recommender")
st.markdown(
    """
    Welcome to the Movie Recommender! Select a genre from the dropdown below
    to get the top 5 movie recommendations based on average ratings from the MovieLens 100k dataset.
    """
)

# Load data
movies_df, all_genres = load_data()

if movies_df is not None: # Only proceed if data loaded successfully
    # Create a dropdown for genre selection
    selected_genre = st.selectbox(
        "Choose a Genre:",
        options=['-- Select a Genre --'] + sorted(all_genres) # Add a default option and sort genres
    )

    # Button to trigger recommendations
    if st.button("Get Recommendations"):
        if selected_genre == '-- Select a Genre --':
            st.warning("Please select a genre to get recommendations.")
        else:
            st.subheader(f"Top 5 Movies in {selected_genre} Genre:")
            recommendations = get_recommendations(selected_genre, movies_df)

            if recommendations:
                for i, movie in enumerate(recommendations):
                    st.write(f"{i+1}. {movie}")
            else:
                st.info("No recommendations found for the selected genre.")

st.markdown(
    """
    ---
    *Built with Python and Streamlit using the MovieLens 100k dataset.*
    """
)
