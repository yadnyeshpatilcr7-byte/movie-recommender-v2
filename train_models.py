# train_models.py
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

class EnhancedMovieRecommender:
    def __init__(self, movies_df, ratings_df):
        self.movies = movies_df
        self.ratings = ratings_df
        self._prepare_data()
        self._build_similarity_matrix()
    
    def _prepare_data(self):
        print("Calculating movie statistics...")
        # Calculate movie stats
        self.movie_stats = self.ratings.groupby('movieId').agg({
            'rating': ['count', 'mean']
        }).round(3)
        self.movie_stats.columns = ['rating_count', 'rating_mean']
        
        # Merge with movies
        self.movies_with_stats = self.movies.merge(
            self.movie_stats, on='movieId', how='left'
        )
        
        # Fill missing values
        self.movies_with_stats['rating_count'] = self.movies_with_stats['rating_count'].fillna(0)
        self.movies_with_stats['rating_mean'] = self.movies_with_stats['rating_mean'].fillna(0)
        
        # Enhanced scoring
        self.movies_with_stats['weighted_score'] = (
            self.movies_with_stats['rating_mean'] * 
            np.log1p(self.movies_with_stats['rating_count'])
        )
        
        # Extract year
        self.movies_with_stats['year'] = self.movies_with_stats['title'].str.extract(r'\((\d{4})\)').fillna(0).astype(int)
        
        print("Data preparation completed!")
    
    def _build_similarity_matrix(self):
        print("Building similarity matrix...")
        self.tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = self.tfidf.fit_transform(self.movies_with_stats['genres'])
        self.cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        print("Similarity matrix built!")
    
    def get_mood_recommendations(self, mood, top_n=10):
        mood_filters = {
            'funny': ['Comedy', 'Animation'],
            'romantic': ['Romance', 'Drama'],
            'action': ['Action', 'Adventure', 'Thriller'],
            'drama': ['Drama', 'Romance'],
            'scary': ['Horror', 'Thriller'],
            'mind_bending': ['Sci-Fi', 'Mystery', 'Thriller'],
            'family': ['Children', 'Family', 'Animation'],
            'inspirational': ['Drama', 'Biography']
        }
        
        if mood not in mood_filters:
            return self.get_popular_movies(top_n)
        
        target_genres = mood_filters[mood]
        mood_movies = self.movies_with_stats[
            self.movies_with_stats['genres'].str.contains('|'.join(target_genres))
        ]
        
        popular_mood_movies = mood_movies[
            (mood_movies['rating_count'] >= 10) & 
            (mood_movies['rating_mean'] >= 3.0)
        ].sort_values('weighted_score', ascending=False)
        
        return popular_mood_movies.head(top_n)
    
    def get_similar_movies(self, movie_title, top_n=10):
        if movie_title not in self.movies_with_stats['title'].values:
            return None
        
        idx = self.movies_with_stats[self.movies_with_stats['title'] == movie_title].index[0]
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n+1]
        movie_indices = [i[0] for i in sim_scores]
        
        return self.movies_with_stats.iloc[movie_indices]
    
    def get_popular_movies(self, top_n=10, min_ratings=50):
        popular = self.movies_with_stats[
            self.movies_with_stats['rating_count'] >= min_ratings
        ].sort_values('weighted_score', ascending=False)
        
        return popular.head(top_n)
    
    def hybrid_recommendations(self, genres=None, min_rating=3.0, min_ratings=10, top_n=15):
        recommendations = self.movies_with_stats.copy()
        
        if genres:
            genre_filter = '|'.join(genres)
            recommendations = recommendations[
                recommendations['genres'].str.contains(genre_filter)
            ]
        
        recommendations = recommendations[
            (recommendations['rating_mean'] >= min_rating) &
            (recommendations['rating_count'] >= min_ratings)
        ].sort_values('weighted_score', ascending=False)
        
        return recommendations.head(top_n)

def train_and_save_models():
    """Train models and save them as PKL files"""
    print("Loading datasets...")
    
    # Load data
    movies = pd.read_csv('data/movies.csv')
    ratings = pd.read_csv('data/ratings.csv')
    
    print(f"Movies: {movies.shape}")
    print(f"Ratings: {ratings.shape}")
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Train recommender
    print("Training enhanced recommender...")
    recommender = EnhancedMovieRecommender(movies, ratings)
    
    # Test the model
    print("\nTesting model...")
    print("Popular movies sample:")
    popular = recommender.get_popular_movies(3)
    for _, movie in popular.iterrows():
        print(f"  - {movie['title']} (Rating: {movie['rating_mean']})")
    
    # Save all models as PKL files
    print("\nSaving models...")
    
    # Save individual components
    with open('models/tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(recommender.tfidf, f)
    
    with open('models/cosine_similarity.pkl', 'wb') as f:
        pickle.dump(recommender.cosine_sim, f)
    
    with open('models/movies_with_stats.pkl', 'wb') as f:
        pickle.dump(recommender.movies_with_stats, f)
    
    # Save the complete recommender
    with open('models/full_recommender.pkl', 'wb') as f:
        pickle.dump(recommender, f)
    
    print("✅ All models saved successfully!")
    print("📁 Files created in 'models/' folder:")
    print("   - tfidf_vectorizer.pkl")
    print("   - cosine_similarity.pkl") 
    print("   - movies_with_stats.pkl")
    print("   - full_recommender.pkl")
    
    return recommender

if __name__ == "__main__":
    print("=" * 60)
    print("MOVIE RECOMMENDER - MODEL TRAINING")
    print("=" * 60)
    
    recommender = train_and_save_models()
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED!")
    print("=" * 60)
    print("\nNext: Run 'app.py' to start the Streamlit app!")