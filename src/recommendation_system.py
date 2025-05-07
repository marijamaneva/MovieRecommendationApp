import chromadb
from chromadb.utils import embedding_functions
import json
import os
import re
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from tmdb_api_helper import TMDBHelper


class MovieRecommender:
    def __init__(self):
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(path="data/embeddings")

        # Get the collection
        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name='all-MiniLM-L6-v2'
        )
        self.collection = self.chroma_client.get_collection(
            name="movie_collection",
            embedding_function=embedding_function
        )

        # Initialize the language model and memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            input_key="human_input"
        )
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, max_tokens=1024)


        # User preferences storage
        self.user_preferences_file = "data/user_preferences.json"
        self.user_preferences = self._load_user_preferences()
        
        # Initialize TMDB helper
        self.tmdb_helper = TMDBHelper()

        # Setup prompt templates
        self._setup_prompts()

    def _setup_prompts(self):
        self.recommendation_template = """
        You are MovieMind, a friendly movie recommendation assistant.

        Previous conversation:
        {chat_history}

        User query: {human_input}

        Based on the query, these are the most relevant movies from our database:
        {movie_results}

        User preferences: {user_preferences}

        Your task:
        1. Analyze the user query and the provided movie results.
        2. Consider the user's preferences if available.
        3. Provide personalized movie recommendations from the list above.
        4. For each recommendation, include:
            - Title: (Movie Title)
            - Year: (Year)
            - Genre: (Genre)
            - Director: (Director)
            - Main actors: (Actors)
            - Short plot summary
            - Reason why you recommend it.
        5. Always provide this information in the exact format shown above.
        6. Make sure each movie is clearly separated from others.
        7. If appropriate, ask a follow-up question to refine future recommendations.

        Give at least 5 movie recommendations, not just a single movie.
        Respond in a conversational, helpful tone. Avoid phrases like "based on your query" or "personalized recommendations".

        AI Assistant:
        """

        self.recommendation_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                input_variables=["chat_history", "human_input", "movie_results", "user_preferences"],
                template=self.recommendation_template
            ),
            memory=self.memory,
            verbose=False
        )

        self.general_template = """
        You are MovieMind, a friendly movie recommendation assistant.

        Previous conversation:
        {chat_history}

        Human: {human_input}

        Respond in a conversational, helpful tone.
        If the user is asking about movies or for recommendations,
        suggest they try asking more specifically about genres, actors, or the type of movie they're looking for.

        AI Assistant:
        """

        self.general_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                input_variables=["chat_history", "human_input"],
                template=self.general_template
            ),
            memory=self.memory,
            verbose=False
        )

    def _load_user_preferences(self):
        """Load user preferences from file"""
        if os.path.exists(self.user_preferences_file):
            with open(self.user_preferences_file, 'r') as f:
                return json.load(f)
        else:
            os.makedirs('data', exist_ok=True)
            with open(self.user_preferences_file, 'w') as f:
                json.dump({}, f)
            return {}

    def _save_user_preferences(self):
        """Save user preferences to file"""
        with open(self.user_preferences_file, 'w') as f:
            json.dump(self.user_preferences, f, indent=2)

    """"
    def update_preferences(self, user_id, movie_id, liked=True):

        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = {"favorites": []}

        prefs = self.user_preferences[user_id]

        self._save_user_preferences()
        """

    def get_response(self, user_id, message):
        """Generate a recommendation or general response based on user input"""
        # Step 1: Search for relevant movies
        results = self.collection.query(
            query_texts=[message],
            n_results=5
        )

        movie_results = results.get("documents", [[]])[0]

        # Step 2: Prepare movie descriptions
        movie_descriptions = ""
        for movie in movie_results:
            if isinstance(movie, str) and movie.strip():
                try:
                    movie_info = json.loads(movie)
                except json.JSONDecodeError:
                    continue
            else:
                movie_info = movie

            if movie_info:
                movie_descriptions += f"\nTitle: {movie_info.get('title', 'Unknown')}\n"
                movie_descriptions += f"Year: {movie_info.get('year', 'Unknown')}\n"
                movie_descriptions += f"Genre: {movie_info.get('genre', 'Unknown')}\n"
                movie_descriptions += f"Director: {movie_info.get('director', 'Unknown')}\n"
                actors = movie_info.get('actors', [])
                if isinstance(actors, list):
                    movie_descriptions += f"Actors: {', '.join(actors)}\n"
                else:
                    movie_descriptions += f"Actors: {actors}\n"
                movie_descriptions += f"Plot: {movie_info.get('plot', 'No plot available')}\n\n"

        # Step 3: Load user preferences
        user_prefs = self.user_preferences.get(user_id, {})
        favorites = user_prefs.get("favorites", [])
        
        if favorites:
            user_preferences_string = (
                f"Favorite movies: {', '.join(favorites)}."
            )
        else:
            user_preferences_string = "No preferences recorded yet."

        # Step 4: Create final response
        try:
            
            response = self.recommendation_chain.invoke({
                "chat_history": self.memory.buffer,
                "human_input": message,
                "movie_results": movie_descriptions,
                "user_preferences": user_preferences_string
            })
            
            # Extract the text response
            if isinstance(response, dict) and "text" in response:
                response = response["text"]
            
        except Exception as e:
            print(f"Error generating recommendation: {e}")
            # Fallback to general response
            try:
                response = self.general_chain.invoke({
                    "chat_history": self.memory.buffer,
                    "human_input": message
                })
                
                if isinstance(response, dict) and "text" in response:
                    response = response["text"]
                    
            except Exception as e2:
                print(f"Error generating general response: {e2}")
                response = "I'm having trouble generating a recommendation right now. Could you try again or ask in a different way?"

        # Step 5: Process the response to add poster data
        processed_response = self.process_response_with_posters(response)
        return processed_response
        
    def process_response_with_posters(self, response):
        """Process the response to add movie poster data"""
        # Split the response into paragraphs
        paragraphs = response.split("\n\n")
        result = ""
        
        # Process each paragraph
        for paragraph in paragraphs:
            # Skip empty paragraphs
            if not paragraph.strip():
                continue
                
            # Check if this paragraph contains movie information
            if "Title:" in paragraph or re.search(r'\b\(\d{4}\)\b', paragraph):
                # Extract movie title
                title_match = re.search(r"Title:\s*(.*?)(?:\n|$)", paragraph)
                if not title_match:
                    # Try to find title in format "Movie Title (Year)"
                    title_match = re.search(r"(.*?)\s*\(\d{4}\)", paragraph)
                
                if title_match:
                    movie_title = title_match.group(1).strip()
                    
                    # Extract year if available
                    year_match = re.search(r"Year:\s*(\d{4})", paragraph)
                    year = year_match.group(1) if year_match else None
                    
                    if not year:
                        # Try to find year in format "Movie Title (Year)"
                        year_match = re.search(r"\((\d{4})\)", paragraph)
                        year = year_match.group(1) if year_match else None
                    
                    # Get poster URL
                    poster_url = self.tmdb_helper.get_poster_url(movie_title, year)
                    
                    # Add poster URL to the response
                    if poster_url:
                        result += paragraph + f"\n[POSTER_URL: {poster_url}]\n\n"
                    else:
                        result += paragraph + "\n\n"
                else:
                    result += paragraph + "\n\n"
            else:
                result += paragraph + "\n\n"
        
        return result