from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the pre-trained model and data
model = pickle.load(open('Model/model.pkl', 'rb'))
book_names = pickle.load(open('Model/book_names.pkl', 'rb'))
final_rating = pickle.load(open('Model/final_rating.pkl', 'rb'))
book_pivot = pickle.load(open('Model/book_pivot.pkl', 'rb'))

# Convert book names to a list for JSON serialization
book_names_list = book_pivot.index.tolist()  # Convert to list of book names

@app.route('/')
def index():
    return render_template('index.html', book_names=book_names_list)

@app.route('/recommend', methods=['POST'])
def recommend():
    # Get the selected book name from the form
    book_name = request.form.get('book_name')

    # Get the book recommendations
    recommended_books, poster_urls = recommend_books(book_name)

    return jsonify({
        'recommended_books': recommended_books,
        'poster_urls': poster_urls
    })

def recommend_books(book_name):
    # Find the book id and get recommendations
    book_id = np.where(book_pivot.index == book_name)[0][0]
    distance, suggestion = model.kneighbors(book_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=6)

    recommended_books = [book_pivot.index[i] for i in suggestion[0]]
    poster_urls = fetch_poster(suggestion)

    return recommended_books, poster_urls

def fetch_poster(suggestion):
    poster_urls = []
    for book_id in suggestion[0]:
        book_name = book_pivot.index[book_id]
        idx = np.where(final_rating['title'] == book_name)[0][0]
        poster_urls.append(final_rating.iloc[idx]['image_url'])
    return poster_urls

if __name__ == "__main__":
    app.run(debug=True)
