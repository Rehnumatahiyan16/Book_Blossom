import pickle
import numpy as np

# Load the pre-trained model and data
model = pickle.load(open('Model/model.pkl', 'rb'))
book_names = pickle.load(open('Model/book_names.pkl', 'rb'))
final_rating = pickle.load(open('Model/final_rating.pkl', 'rb'))
book_pivot = pickle.load(open('Model/book_pivot.pkl', 'rb'))

# Function to fetch poster URLs for recommended books
def fetch_poster(suggestion):
    book_name = [book_pivot.index[i] for i in suggestion.flatten()]  # Ensure we're getting the actual names
    poster_url = []

    # Fetch URLs
    for name in book_name:
        matches = final_rating[final_rating['title'] == name]  # Filter DataFrame for the title
        if not matches.empty:
            url = matches.iloc[0]['image_url']  # Safely get the first match
            poster_url.append(url)
        else:
            poster_url.append("No image available")  # Handle missing images

    return poster_url

# Function to recommend books based on the input book name
def recommend_book(book_name):
    try:
        # Find the index of the selected book in the pivot table
        book_id = np.where(book_pivot.index == book_name)[0][0]
        # Get recommendations from the model
        distances, suggestions = model.kneighbors(book_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=6)

        # Fetch poster URLs for recommended books
        poster_urls = fetch_poster(suggestions[0])  # Pass the correct index array

        # Collect recommended books
        recommended_books = [book_pivot.index[i] for i in suggestions[0]]
        return recommended_books, poster_urls
    except IndexError:
        print("Book not found in the model.")
        return [], []

# Main function to interact with the user
def main():
    print("Welcome to the Book Recommender System!")
    
    # Display available books
    print("\nAvailable books:")
    for idx, book in enumerate(book_names[:30], 1):  # Display only the first 30 books for convenience
        print(f"{idx}. {book}")
    
    # Get user input
    selected_book_index = int(input("\nEnter the number of a book you'd like to get recommendations for: ")) - 1
    selected_book = book_names[selected_book_index]
    
    print(f"\nYou selected: {selected_book}")
    
    # Get recommendations
    recommended_books, poster_urls = recommend_book(selected_book)

    if recommended_books:
        print("\nRecommended Books:")
        for i in range(len(recommended_books)):
            print(f"{i+1}. {recommended_books[i]}")

            if i < len(poster_urls):
                print(f"Poster: {poster_urls[i]}")
    else:
        print("\nNo recommendations available.")

# Run the main function
if __name__ == "__main__":
    main()
