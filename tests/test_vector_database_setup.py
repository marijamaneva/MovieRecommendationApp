import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from src.vector_database_setup import prepare_movie_descriptions, create_vector_database

# Sample DataFrame to use in both tests
@pytest.fixture
def sample_movies_df():
    return pd.DataFrame({
        'movieId': [1],
        'title': ['The Matrix (1999)'],
        'clean_title': ['The Matrix'],
        'year': ['1999'],
        'genres': [['Action', 'Sci-Fi']],
        'avg_rating': [4.5],
        'rating_count': [1200]
    })


def test_prepare_movie_descriptions(sample_movies_df):
    updated_df = prepare_movie_descriptions(sample_movies_df.copy())
    
    assert 'description' in updated_df.columns
    assert "The Matrix" in updated_df['description'].iloc[0]
    assert "1999" in updated_df['description'].iloc[0]
    assert "Action" in updated_df['description'].iloc[0]


@patch("src.vector_database_setup.chromadb.PersistentClient")
@patch("src.vector_database_setup.embedding_functions.SentenceTransformerEmbeddingFunction")
@patch("src.vector_database_setup.SentenceTransformer")
def test_create_vector_database(mock_model, mock_embed_func, mock_client, sample_movies_df):
    # Add 'description' column just like prepare_movie_descriptions would
    sample_movies_df['description'] = sample_movies_df.apply(
        lambda row: f"Title: {row['clean_title']}. "
                    f"Year: {row['year']}. "
                    f"Genres: {', '.join(row['genres'])}. "
                    f"Average Rating: {row['avg_rating']:.1f}/5 from {int(row['rating_count'])} users.",
        axis=1
    )

    # Mock Chroma collection and client
    mock_collection = MagicMock()
    mock_client.return_value.create_collection.return_value = mock_collection

    # Run function
    collection = create_vector_database(sample_movies_df)

    # Assertions
    mock_client.return_value.create_collection.assert_called_once()
    mock_collection.add.assert_called_once()
    assert collection == mock_collection
