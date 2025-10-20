import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load ratings and movie titles
ratings = pd.read_csv('ml-100k\\u.data', sep='\t', names=['user_id','movie_id','rating','timestamp'])
movies = pd.read_csv('ml-100k\\u.item', sep='|', encoding='latin-1', header=None, usecols=[0,1], names=['movie_id','title'])

# Merge and pivot into a user-movie matrix
ratings = ratings.merge(movies, on='movie_id')
user_movie_matrix = ratings.pivot_table(index='user_id', columns='title', values='rating').fillna(0)

# Compute item-to-item cosine similarity
similarity_matrix = cosine_similarity(user_movie_matrix.T)
similarity_df = pd.DataFrame(similarity_matrix, index=user_movie_matrix.columns, columns=user_movie_matrix.columns)

def get_recommendations(title, n=5):
    """
    Given a movie title, return a list of n similar movie titles.
    """
    if title not in similarity_df.index:
        return None
    # sort descending, skip the first (it's the same movie)
    similar_scores = similarity_df[title].sort_values(ascending=False)
    return list(similar_scores.iloc[1:n+1].index)

def chat():
    print("Hello! I can suggest movies similar to ones you like.")
    print("Type the exact movie title from the MovieLens dataset, or type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit']:
            print("Agent: Goodbye!")
            break
        recs = get_recommendations(user_input)
        if recs is None:
            print("Agent: Sorry, I don't recognize that title. Please try another.")
        else:
            print("Agent: Since you like {}, you might also enjoy: {}".format(user_input, ', '.join(recs)))

if __name__ == '__main__':
    chat()
