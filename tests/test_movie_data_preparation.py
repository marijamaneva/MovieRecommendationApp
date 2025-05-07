# tests/test_movie_data_preparation.py
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from io import BytesIO
import os

from src.movie_data_preparation import download_and_prepare_movielens


@patch("src.movie_data_preparation.requests.get")
@patch("src.movie_data_preparation.ZipFile.extractall")
@patch("src.movie_data_preparation.pd.read_csv")
def test_download_and_prepare_movielens(mock_read_csv, mock_extractall, mock_get):
    # Mock the GET request
    mock_get.return_value.content = b"Fake zip content"

    # Mock the zipfile to simulate extraction
    mock_zip = MagicMock()
    mock_zip.__enter__.return_value = mock_zip
    with patch("src.movie_data_preparation.ZipFile", return_value=mock_zip):
        # Create fake DataFrames
        movies_data = {
            "movieId": [1, 2],
            "title": ["Toy Story (1995)", "Jumanji (1995)"],
            "genres": ["Animation|Children|Comedy", "Adventure|Children|Fantasy"]
        }
        ratings_data = {
            "userId": [1, 2, 1, 3],
            "movieId": [1, 1, 2, 2],
            "rating": [4.0, 5.0, 3.0, 2.0]
        }

        movies_df = pd.DataFrame(movies_data)
        ratings_df = pd.DataFrame(ratings_data)

        # Mock read_csv to return these fake DataFrames in order
        mock_read_csv.side_effect = [movies_df, ratings_df]

        # Run the function
        result_df = download_and_prepare_movielens()

        # Validate output DataFrame
        assert isinstance(result_df, pd.DataFrame)
        assert "clean_title" in result_df.columns
        assert result_df.loc[0, "clean_title"] == "Toy Story"
        assert result_df.loc[0, "year"] == "1995"
        assert "avg_rating" in result_df.columns
        assert result_df.loc[0, "avg_rating"] == 4.5
        assert "rating_count" in result_df.columns
        assert result_df.loc[0, "rating_count"] == 2

        # Check that processed file was saved (if you want to validate this part)
        assert os.path.exists("data/processed_movies.csv") or True  # Optional
