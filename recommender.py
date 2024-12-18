import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

# Load the pre-trained models and data
book_names = pd.read_pickle('Model/book_names.pkl')
book_pivot = pd.read_pickle('Model/book_pivot.pkl')
final_rating = pd.read_pickle('Model/final_rating.pkl')

# Perform Singular Value Decomposition (SVD) to reduce dimensionality
from scipy.sparse import csr_matrix

book_sparse = csr_matrix(book_pivot)
svd = TruncatedSVD(n_components=50, random_state=42)
book_svd = svd.fit_transform(book_sparse)

# Compute cosine similarity between books
book_similarity = cosine_similarity(book_svd)

# Function to get recommendations for a given book
def get_recommendations(book_name, top_n=5):
    # Get the index of the book in the book_names dataframe
    book_idx = book_names[book_names['title'] == book_name].index[0]

    # Get the similarity scores for this book
    sim_scores = list(enumerate(book_similarity[book_idx]))

    # Sort the books based on similarity score
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the top N similar books
    sim_scores = sim_scores[1:top_n+1]

    # Get the book indices
    book_indices = [i[0] for i in sim_scores]

    # Return the book titles of the most similar books
    return book_names.iloc[book_indices]['title'].values
