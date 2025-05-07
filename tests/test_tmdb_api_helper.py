import pytest
from unittest.mock import patch
from src.tmdb_api_helper import TMDBHelper

@pytest.fixture
def tmdb_helper():
    return TMDBHelper()

@pytest.fixture
def mock_tmdb_response():
    return {
        "results": [
            {
                "id": 12345,
                "title": "Test Movie",
                "poster_path": "/poster.jpg",
                "overview": "A test movie.",
                "release_date": "2023-01-01"
            }
        ]
    }

def test_search_movie_success(tmdb_helper, mock_tmdb_response):
    """Test movie search with valid response"""
    with patch('src.tmdb_api_helper.requests.get') as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = mock_tmdb_response

        result = tmdb_helper.search_movie("Test Movie", "2023")

        assert result is not None
        assert result['title'] == "Test Movie"
        assert result['poster_path'] == "https://image.tmdb.org/t/p/w500/poster.jpg"

def test_search_movie_failure(tmdb_helper):
    """Test search when API key is missing"""
    tmdb_helper.api_key = None
    result = tmdb_helper.search_movie("Test Movie")
    assert result is None

def test_get_poster_url_success(tmdb_helper):
    """Test get_poster_url returns correct poster URL"""
    with patch.object(tmdb_helper, 'search_movie') as mock_search:
        mock_search.return_value = {
            'poster_path': "https://image.tmdb.org/t/p/w500/poster.jpg"
        }

        url = tmdb_helper.get_poster_url("Test Movie")
        assert url == "https://image.tmdb.org/t/p/w500/poster.jpg"

def test_get_poster_url_no_poster(tmdb_helper):
    """Test get_poster_url when no poster is found"""
    with patch.object(tmdb_helper, 'search_movie') as mock_search:
        mock_search.return_value = {"poster_path": None}
        assert tmdb_helper.get_poster_url("Test Movie") is None
