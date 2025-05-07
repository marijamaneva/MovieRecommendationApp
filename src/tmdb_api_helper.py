import requests
import os
import time
from dotenv import load_dotenv

class TMDBHelper:
    def __init__(self):
        load_dotenv()
        self.api_key = os.environ.get("TMDB_API_KEY")
        self.base_url = "https://api.themoviedb.org/3"
        self.poster_base_url = "https://image.tmdb.org/t/p/w500"
        self.search_cache = {}  # Simple cache to avoid repeated API calls
        self.last_request_time = 0  # For rate limiting
        
        if not self.api_key:
            print("Warning: TMDB_API_KEY not found in environment variables.")
            print("Please set it in your .env file to enable movie posters.")
    
    def _rate_limit(self):
        """Implement simple rate limiting to avoid API restrictions"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        # Wait if less than 0.25 seconds since last request (4 requests per second)
        if time_since_last_request < 0.25:
            time.sleep(0.25 - time_since_last_request)
            
        self.last_request_time = time.time()
    
    def search_movie(self, title, year=None):
        """Search for a movie by title and optional year"""
        if not self.api_key:
            return None
        
        # Check cache first
        cache_key = f"{title}_{year}"
        if cache_key in self.search_cache:
            return self.search_cache[cache_key]
            
        # Rate limit API calls
        self._rate_limit()
        
        try:
            url = f"{self.base_url}/search/movie"
            params = {
                "api_key": self.api_key,
                "query": title,
                "include_adult": "false"
            }
            
            if year:
                params["year"] = year
                
            response = requests.get(url, params=params, timeout=5)
            
            if response.status_code == 200:
                results = response.json().get("results", [])
                if results:
                    # Return the first result
                    movie_data = results[0]
                    result = {
                        "id": movie_data.get("id"),
                        "title": movie_data.get("title"),
                        "poster_path": f"{self.poster_base_url}{movie_data.get('poster_path')}" if movie_data.get('poster_path') else None,
                        "overview": movie_data.get("overview"),
                        "release_date": movie_data.get("release_date")
                    }
                    
                    # Cache the result
                    self.search_cache[cache_key] = result
                    return result
                    
        except Exception as e:
            print(f"Error searching movie: {e}")
            
        return None
    
    def get_poster_url(self, movie_title, year=None):
        """Get poster URL for a movie title"""
        if not movie_title:
            return None
            
        try:
            movie_data = self.search_movie(movie_title, year)
            if movie_data and movie_data.get("poster_path"):
                return movie_data.get("poster_path")
        except Exception as e:
            print(f"Error getting poster URL: {e}")
            
        return None