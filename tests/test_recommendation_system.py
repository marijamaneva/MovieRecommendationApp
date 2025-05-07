import unittest
from unittest.mock import patch, MagicMock, mock_open
import json
import os
import sys

# Mock all external dependencies that might cause import issues
sys.modules['tmdb_api_helper'] = MagicMock()
sys.modules['chromadb'] = MagicMock()
sys.modules['chromadb.utils'] = MagicMock()
sys.modules['chromadb.utils.embedding_functions'] = MagicMock()
sys.modules['langchain_openai'] = MagicMock()
sys.modules['langchain.prompts'] = MagicMock()
sys.modules['langchain.chains'] = MagicMock()
sys.modules['langchain.memory'] = MagicMock()

# Create mock classes for each imported class
mock_classes = {
    'TMDBHelper': MagicMock(),
    'OpenAI': MagicMock(),
    'PromptTemplate': MagicMock(),
    'LLMChain': MagicMock(),
    'ConversationBufferMemory': MagicMock(),
    'PersistentClient': MagicMock(),
    'SentenceTransformerEmbeddingFunction': MagicMock()
}

# Make sure each mock is properly set in its module
sys.modules['tmdb_api_helper'].TMDBHelper = mock_classes['TMDBHelper']
sys.modules['langchain_openai'].OpenAI = mock_classes['OpenAI']
sys.modules['langchain.prompts'].PromptTemplate = mock_classes['PromptTemplate']
sys.modules['langchain.chains'].LLMChain = mock_classes['LLMChain']
sys.modules['langchain.memory'].ConversationBufferMemory = mock_classes['ConversationBufferMemory']
sys.modules['chromadb'].PersistentClient = mock_classes['PersistentClient']
sys.modules['chromadb.utils.embedding_functions'].SentenceTransformerEmbeddingFunction = mock_classes['SentenceTransformerEmbeddingFunction']

# Now define a test class to import MovieRecommender
class MovieRecommenderTests(unittest.TestCase):
    
    @patch('builtins.open', new_callable=mock_open, read_data='{}')
    @patch('os.path.exists', return_value=True)
    @patch('os.makedirs')
    def setUp(self, mock_makedirs, mock_exists, mock_file):
        """Import MovieRecommender dynamically to avoid import errors at module level"""
        # Add project root to path
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        
        # Import module dynamically after all mocks are in place
        from src.recommendation_system import MovieRecommender
        
        # Create instance
        self.recommender = MovieRecommender()
        
        # Setup additional mocks
        self.mock_collection = MagicMock()
        mock_classes['PersistentClient'].return_value.get_collection.return_value = self.mock_collection
        
        # Setup mock for TMDBHelper instance
        mock_classes['TMDBHelper'].return_value.get_poster_url.return_value = "http://example.com/poster.jpg"
        
        # Create a placeholder for user preferences
        self.recommender.user_preferences = {}
        
        # Mock save_user_preferences method to avoid file operations
        self.recommender._save_user_preferences = MagicMock()

    def test_update_preferences_new_user(self):
        """Test updating preferences for a new user"""
        # Call the method
        self.recommender.update_preferences("test_user", "movie123", liked=True)
        
        # Verify user was created with correct structure
        self.assertIn("test_user", self.recommender.user_preferences)
        self.assertIn("liked", self.recommender.user_preferences["test_user"])
        self.assertIn("disliked", self.recommender.user_preferences["test_user"])
        self.assertIn("favorites", self.recommender.user_preferences["test_user"])
        
        # Check movie was added to correct lists
        self.assertIn("movie123", self.recommender.user_preferences["test_user"]["liked"])
        self.assertIn("movie123", self.recommender.user_preferences["test_user"]["favorites"])
        self.assertNotIn("movie123", self.recommender.user_preferences["test_user"]["disliked"])
        
        # Verify save was called
        self.recommender._save_user_preferences.assert_called_once()
    
    def test_update_preferences_dislike(self):
        """Test updating preferences to dislike a movie"""
        # Setup existing user
        self.recommender.user_preferences = {
            "test_user": {
                "liked": [],
                "disliked": [],
                "favorites": []
            }
        }
        
        # Call the method with liked=False
        self.recommender.update_preferences("test_user", "movie123", liked=False)
        
        # Check movie was added to disliked list only
        self.assertIn("movie123", self.recommender.user_preferences["test_user"]["disliked"])
        self.assertNotIn("movie123", self.recommender.user_preferences["test_user"]["liked"])
        self.assertNotIn("movie123", self.recommender.user_preferences["test_user"]["favorites"])
        
        # Verify save was called
        self.recommender._save_user_preferences.assert_called_once()

if __name__ == '__main__':
    unittest.main()