import json
import pandas as pd
import os
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions

def prepare_movie_descriptions(movies_df):
    """
    Prepare JSON document strings for each movie (format expected by recommendation_system.py).
    Also adds a plain-text 'description' column used as the embedding source.
    """
    def make_doc(row):
        genres = row['genres'] if isinstance(row['genres'], list) else []
        year = str(row['year']) if pd.notna(row['year']) else "Unknown"
        genre_str = ', '.join(genres) if genres else "Unknown"
        plot = (
            f"A {genre_str} film from {year}. "
            f"Rated {row['avg_rating']:.1f}/5 by {int(row['rating_count'])} users."
        )
        doc = {
            "title": row['clean_title'],
            "year": year,
            "genre": genre_str,
            "director": "Unknown",
            "actors": [],
            "plot": plot,
        }
        return json.dumps(doc)

    movies_df['description'] = movies_df.apply(make_doc, axis=1)
    return movies_df

def create_vector_database(movies_df):
    """
    Create a Chroma vector database with movie embeddings
    """
    print("Creating vector database...")
    
    # Create embeddings directory if it doesn't exist
    if not os.path.exists('data/embeddings'):
        os.makedirs('data/embeddings')
    
    # Initialize the sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Initialize ChromaDB client
    chroma_client = chromadb.PersistentClient(path="data/embeddings")
    
    # Create or get the collection
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name='all-MiniLM-L6-v2'
    )
    
    # Delete collection if it exists (for demo purposes)
    try:
        chroma_client.delete_collection("movie_collection")
    except:
        pass
    
    # Create new collection
    collection = chroma_client.create_collection(
        name="movie_collection",
        embedding_function=embedding_function
    )
    
    # Add movies to the collection in batches
    batch_size = 100
    for i in range(0, len(movies_df), batch_size):
        batch = movies_df.iloc[i:i+batch_size]
        
        collection.add(
            ids=[str(id) for id in batch['movieId'].tolist()],
            documents=batch['description'].tolist(),
            metadatas=[{
                'title': row['clean_title'],
                'year': row['year'],
                'genres': ','.join(row['genres']),
                'avg_rating': str(row['avg_rating']),
                'rating_count': str(row['rating_count'])
            } for _, row in batch.iterrows()]
        )
        
        print(f"Added {i+len(batch)}/{len(movies_df)} movies to vector database")
    
    print("Vector database created successfully!")
    return collection

if __name__ == "__main__":
    movies_df = pd.read_csv('data/processed_movies.csv')
    
    # Convert string representation of list back to list
    movies_df['genres'] = movies_df['genres'].apply(eval)
    
    # Prepare movie descriptions
    movies_df = prepare_movie_descriptions(movies_df)
    
    # Create vector database
    collection = create_vector_database(movies_df)
    
    # Test the database with a query
    results = collection.query(
        query_texts=["action movies with high ratings"],
        n_results=5
    )
    
    print("\nTest query results:")
    for i, (id, document, metadata) in enumerate(zip(
        results['ids'][0], results['documents'][0], results['metadatas'][0]
    )):
        print(f"{i+1}. {metadata['title']} ({metadata['year']}) - {metadata['genres']}")