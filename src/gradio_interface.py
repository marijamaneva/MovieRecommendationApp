import gradio as gr
import os
import dotenv
import json
import re
from pathlib import Path
from recommendation_system import MovieRecommender

# Disable tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables
dotenv.load_dotenv()

# Check if OpenAI API key is set
if "OPENAI_API_KEY" not in os.environ:
    print("Warning: OPENAI_API_KEY not found in environment variables.")
    print("Please set it in a .env file or directly in your environment.")

# Check for TMDB API key
if "TMDB_API_KEY" not in os.environ:
    print("Warning: TMDB_API_KEY not found in environment variables.")
    print("Please set it in a .env file to enable movie posters.")

# Initialize the movie recommender
recommender = MovieRecommender()

# Default user (mocked for demo)
DEFAULT_USER_ID = "demo_user"
PREFERENCES_FILE = "data/user_preferences.json"

# --- FAVORITES LOGIC ---
def load_preferences():
    if Path(PREFERENCES_FILE).exists():
        with open(PREFERENCES_FILE, "r") as f:
            prefs = json.load(f)
    else:
        prefs = {}
    
    # Ensure "favorites" key exists
    if DEFAULT_USER_ID not in prefs:
        prefs[DEFAULT_USER_ID] = {"favorites": []}
    elif "favorites" not in prefs[DEFAULT_USER_ID]:
        prefs[DEFAULT_USER_ID]["favorites"] = []
    return prefs

def save_preferences(prefs):
    with open(PREFERENCES_FILE, "w") as f:
        json.dump(prefs, f, indent=2)

def save_favorite_movie(movie_title):
    prefs = load_preferences()
    if movie_title not in prefs[DEFAULT_USER_ID]["favorites"]:
        prefs[DEFAULT_USER_ID]["favorites"].append(movie_title)
        save_preferences(prefs)
        return f"✅ '{movie_title}' saved to favorites!"
    return f"ℹ️ '{movie_title}' is already in favorites."

def delete_favorite_movie(movie_title):
    prefs = load_preferences()
    if movie_title in prefs[DEFAULT_USER_ID]["favorites"]:
        prefs[DEFAULT_USER_ID]["favorites"].remove(movie_title)
        save_preferences(prefs)
        return f"🗑️ '{movie_title}' removed from favorites."
    return f"⚠️ '{movie_title}' not found in favorites."

def list_favorite_movies():
    prefs = load_preferences()
    if prefs[DEFAULT_USER_ID].get("favorites"):
        return "\n".join(f"- {m}" for m in prefs[DEFAULT_USER_ID]["favorites"])
    return "You have no favorite movies yet."

# --- PROCESS RESPONSE WITH POSTERS ---
def process_response(response):
    """Process the chatbot response to extract movie information and poster URLs"""
    # Extract poster URLs
    poster_pattern = r'\[POSTER_URL: (.*?)\]'
    poster_urls = re.findall(poster_pattern, response)
    
    # If no posters found, return original text
    if not poster_urls:
        return response, ""
    
    # Split the response by poster tags
    parts = re.split(r'\[POSTER_URL: .*?\]', response)
    
    # Clean text response (remove poster tags)
    clean_response = "".join(parts)
    
    # Create HTML output with posters
    html_output = '<div style="display: flex; flex-wrap: wrap; gap: 20px; justify-content: center; margin-top: 20px;">'
    
    # Find movie titles
    movie_titles = []
    all_text = response
    
    # Extract movie titles by finding patterns in the text
    # Look for patterns like "Title: Movie Name" or "Movie Name (Year)"
    title_pattern = r'Title:\s*(.*?)(?:\n|$)'
    year_pattern = r'(\b[A-Z][^()\n]*?)\s*\(\d{4}\)'
    
    title_matches = re.findall(title_pattern, all_text)
    year_matches = re.findall(year_pattern, all_text)
    
    # Combine found titles, prioritizing explicit "Title:" format
    potential_titles = title_matches + year_matches
    
    # If we found fewer titles than posters, use a more aggressive approach
    if len(potential_titles) < len(poster_urls):
        # Try to extract titles from the text chunks between poster URLs
        # This looks for the first line or something that looks like a title
        for i, part in enumerate(parts):
            if i < len(poster_urls):  # Make sure we don't go beyond available posters
                lines = part.strip().split('\n')
                if lines:
                    # Get the first non-empty line as a potential title
                    for line in lines:
                        line = line.strip()
                        if line:
                            # Skip lines that look like descriptions
                            if len(line) < 100 and not line.startswith(("- ", "• ")):
                                potential_titles.append(line)
                                break
    
    # Fill in missing titles if needed
    while len(potential_titles) < len(poster_urls):
        potential_titles.append(f"Movie {len(potential_titles) + 1}")
    
    # Process each poster
    for i in range(len(poster_urls)):
        title = "Movie"  # Default title
        
        # Use the extracted title if available
        if i < len(potential_titles):
            title = potential_titles[i]
        
        

        html_output += f"""
       <div style="text-align: center; max-width: 200px;">
            <div style="color: black; font-weight: bold; margin-bottom: 8px; font-size: 14px; text-shadow: 1px 1px 2px rgba(255,255,255,0.7); height: 40px; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;">
                <strong>{title}</strong>
            </div>
            <img src="{poster_urls[i]}" alt="{title}" style="max-height: 300px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.3);" />
        </div>
        """
    
    html_output += '</div>'
    
    return clean_response, html_output

# --- CHATBOT REPLY ---
def respond(message: str, history: list) -> tuple:
    # Get text response from recommender
    full_response = recommender.get_response(DEFAULT_USER_ID, message)
    
    # Process response to create HTML with posters
    text_response, html_posters = process_response(full_response)
    
    return text_response, html_posters

# --- GRADIO INTERFACE ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.HTML("""
    <div style="text-align: center; margin-bottom: 1rem">
        <h1>MovieMind 🎬</h1>
        <p>Ask for movie recommendations and manage your favorite movies.</p>
    </div>
    """)
    
    chatbot = gr.Chatbot(height=400)
    movie_posters = gr.HTML(label="Movie Posters")
    
    with gr.Row():
        msg = gr.Textbox(
            placeholder="Ask for movie recommendations...",
            show_label=False,
            container=False,
            scale=12  # Increased scale for wider input box
        )
        send_btn = gr.Button("→", elem_id="send-btn", scale=1)
        clear = gr.Button("🗑️", elem_id="trash-btn", scale=1)  # Changed to trash icon
    
    with gr.Row():
        movie_input = gr.Textbox(label="Movie title")
        save_btn = gr.Button("Save to Favorites")
        delete_btn = gr.Button("Delete from Favorites")
        view_btn = gr.Button("View Favorites")
    
    output = gr.Textbox(label="Favorite Movies List", lines=6)
    
    def user(user_message, history):
        return "", history + [[user_message, None]]
    
    def bot(history):
        text_response, html_posters = respond(history[-1][0], history)
        history[-1][1] = text_response
        return history, html_posters
    
    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, [chatbot, movie_posters]
    )
    
    # Add the send button functionality - same as submitting the text input
    send_btn.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, [chatbot, movie_posters]
    )
    
    clear.click(lambda: ([], ""), None, [chatbot, movie_posters], queue=False)
    
    save_btn.click(fn=save_favorite_movie, inputs=movie_input, outputs=output)
    delete_btn.click(fn=delete_favorite_movie, inputs=movie_input, outputs=output)
    view_btn.click(fn=list_favorite_movies, outputs=output)

    # Add CSS to style the buttons and inputs
    gr.HTML("""
    <style>
    /* Square send button */
    #send-btn {
        border-radius: 8px;
        min-width: 40px;
        margin-right: 5px;
    }
    
    /* Trash button styling */
    #trash-btn {    
        border-radius: 8px;
        min-width: 40px;
        margin-right: 5px;
    }
    </style>
    """)

demo.launch(share=True)