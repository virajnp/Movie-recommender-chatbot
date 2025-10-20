from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import re, requests, os
from dotenv import load_dotenv

load_dotenv()  # loads variables from .env into the environment

OMDB_API_KEY = os.getenv('OMDB_API_KEY')

def get_movie_info(title):
    params = {
        't': title,
        'apikey': OMDB_API_KEY,
        'plot': 'short'
    }
    response = requests.get('http://www.omdbapi.com/', params=params)
    data = response.json()
    if data.get('Response') == 'True':
        return {
            'title': data.get('Title'),
            'year': data.get('Year'),
            'rated': data.get('Rated'),
            'genre': data.get('Genre'),
            'director': data.get('Director'),
            'actors': data.get('Actors'),
            'plot': data.get('Plot')
        }
    else:
        return None

# Data loading and similarity computation (unchanged)
ratings = pd.read_csv('ml-100k\\u.data', sep='\t', names=['user_id','movie_id','rating','timestamp'])
movies = pd.read_csv('ml-100k\\u.item', sep='|', encoding='latin-1', header=None, usecols=[0,1], names=['movie_id','title'])
ratings = ratings.merge(movies, on='movie_id')
user_movie_matrix = ratings.pivot_table(index='user_id', columns='title', values='rating').fillna(0)
similarity_matrix = cosine_similarity(user_movie_matrix.T)
similarity_df = pd.DataFrame(similarity_matrix, index=user_movie_matrix.columns, columns=user_movie_matrix.columns)

def get_recommendations(title, n=5):
    if title not in similarity_df.index:
        return None
    similar_scores = similarity_df[title].sort_values(ascending=False)
    return list(similar_scores.iloc[1:n+1].index)

app = Flask(__name__)

@app.route('/chat', methods=['GET'])
def chat_page():
    return render_template('chat.html')

@app.route('/chat', methods=['POST'])
def chat_api():
    data = request.get_json()
    user_message = data.get('message', '').strip()

    # Check for information request
    info_match = re.search(r'(tell me about|what is|what\'s|who starred in)\s+(.+)', user_message, re.IGNORECASE)
    if info_match:
        title = info_match.group(2).strip()
        info = get_movie_info(title)
        if info:
            reply = (f"{info['title']} ({info['year']}, {info['genre']}) — directed by {info['director']}. "
                     f"Cast: {info['actors']}. Plot: {info['plot']}")
        else:
            reply = "Sorry, I couldn't find information on that movie."
        return jsonify({'reply': reply})

    # Check for recommendation request
    rec_match = re.search(r'recommend(?: me)?(?: a movie like)?\s+(.+)', user_message, re.IGNORECASE)
    if rec_match:
        title = rec_match.group(1).strip()
        recs = get_recommendations(title)
        if recs:
            reply = f'Since you like {title}, you might also enjoy: ' + ', '.join(recs)
        else:
            reply = "Sorry, I couldn't find that title. Try another one."
        return jsonify({'reply': reply})

    # Default response
    return jsonify({'reply': "I'm here to recommend movies and tell you about them.  Ask me to recommend a movie or to tell you about one."})

@app.route('/', methods=['GET'])
def home():
    # Redirect to chat page for convenience
    return render_template('chat.html')

if __name__ == '__main__':
    app.run(debug=True)
