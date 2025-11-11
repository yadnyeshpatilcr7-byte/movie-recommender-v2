# train_models.py - KNN + Random Forest Hybrid (VS Code Version)
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class EnhancedMovieRecommender:
    def __init__(self, movies_df, ratings_df, tags_df=None):
        self.movies = movies_df
        self.ratings = ratings_df
        self.tags = tags_df
        self._prepare_data()
        self._build_similarity_matrix()
        self._train_knn()
        self._train_random_forest()
    
    def _prepare_data(self):
        print("Calculating movie statistics...")
        # Calculate movie stats
        self.movie_stats = self.ratings.groupby('movieId').agg({
            'rating': ['count', 'mean']
        }).round(3)
        self.movie_stats.columns = ['rating_count', 'rating_mean']
        
        # Process tags if available
        if self.tags is not None:
            movie_tags = self.tags.groupby('movieId')['tag'].apply(
                lambda x: ' '.join(x) if len(x) > 0 else ''
            ).reset_index()
            movie_tags.columns = ['movieId', 'all_tags']
        else:
            movie_tags = pd.DataFrame({'movieId': self.movies['movieId'], 'all_tags': ''})
        
        # Merge with movies
        self.movies_with_stats = self.movies.merge(
            self.movie_stats, on='movieId', how='left'
        ).merge(
            movie_tags, on='movieId', how='left'
        )
        
        # Fill missing values
        self.movies_with_stats['rating_count'] = self.movies_with_stats['rating_count'].fillna(0)
        self.movies_with_stats['rating_mean'] = self.movies_with_stats['rating_mean'].fillna(0)
        self.movies_with_stats['all_tags'] = self.movies_with_stats['all_tags'].fillna('')
        
        # Enhanced scoring
        self.movies_with_stats['weighted_score'] = (
            self.movies_with_stats['rating_mean'] * 
            np.log1p(self.movies_with_stats['rating_count'])
        )
        
        # Extract year
        self.movies_with_stats['year'] = self.movies_with_stats['title'].str.extract(r'\((\d{4})\)').fillna(0).astype(int)
        
        # Combine genres and tags for better content analysis
        self.movies_with_stats['genres_tags'] = (
            self.movies_with_stats['genres'] + ' ' + self.movies_with_stats['all_tags']
        )
        
        # Prepare features for Random Forest
        self._prepare_rf_features()
        
        print("Data preparation completed!")
    
    def _prepare_rf_features(self):
        """Prepare features for Random Forest training"""
        # Merge ratings with movie features
        self.ratings_with_features = self.ratings.merge(
            self.movies_with_stats, on='movieId', how='left'
        )
        
        # Feature engineering for Random Forest
        self.ratings_with_features['user_avg_rating'] = self.ratings_with_features.groupby('userId')['rating'].transform('mean')
        self.ratings_with_features['movie_avg_rating'] = self.ratings_with_features.groupby('movieId')['rating'].transform('mean')
        
        # Calculate tag diversity if tags available
        if self.tags is not None:
            tag_counts = self.tags.groupby('movieId')['tag'].nunique().reset_index()
            tag_counts.columns = ['movieId', 'tag_diversity']
            self.ratings_with_features = self.ratings_with_features.merge(
                tag_counts, on='movieId', how='left'
            )
            self.ratings_with_features['tag_diversity'] = self.ratings_with_features['tag_diversity'].fillna(0)
            self.feature_columns = [
                'user_avg_rating', 'movie_avg_rating', 'rating_count', 
                'rating_mean', 'year', 'tag_diversity'
            ]
        else:
            self.ratings_with_features['tag_diversity'] = 0
            self.feature_columns = [
                'user_avg_rating', 'movie_avg_rating', 'rating_count', 
                'rating_mean', 'year'
            ]
    
    def _build_similarity_matrix(self):
        print("Building enhanced similarity matrix...")
        self.tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        tfidf_matrix = self.tfidf.fit_transform(self.movies_with_stats['genres_tags'])
        self.cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        print("Similarity matrix built!")
    
    def _train_knn(self):
        print("Training KNN model...")
        # Build user-movie matrix for collaborative filtering
        self.user_movie_matrix = self.ratings.pivot(
            index='userId', 
            columns='movieId', 
            values='rating'
        ).fillna(0)
        
        # Train KNN model
        self.knn = NearestNeighbors(
            n_neighbors=20, 
            metric='cosine',
            algorithm='brute'
        )
        self.knn.fit(self.user_movie_matrix)
        print("KNN model trained!")
    
    def _train_random_forest(self):
        print("Training Random Forest model...")
        
        # Prepare training data
        X = self.ratings_with_features[self.feature_columns].fillna(0)
        y = self.ratings_with_features['rating']
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train Random Forest
        self.rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        )
        
        self.rf_model.fit(self.X_train, self.y_train)
        
        # Make predictions and evaluate
        self.y_pred = self.rf_model.predict(self.X_test)
        
        # Calculate metrics
        rf_mae = mean_absolute_error(self.y_test, self.y_pred)
        rf_rmse = np.sqrt(mean_squared_error(self.y_test, self.y_pred))
        
        print(f"Random Forest Performance:")
        print(f"  MAE:  {rf_mae:.3f}")
        print(f"  RMSE: {rf_rmse:.3f}")
        print("Random Forest model trained!")
    
    def evaluate_knn(self):
        """Evaluate KNN using hit rate"""
        print("Evaluating KNN model...")
        hits = 0
        total = 0
        
        # Sample users for evaluation
        test_users = self.ratings['userId'].unique()[:50]
        
        for user_id in test_users:
            user_ratings = self.ratings[self.ratings['userId'] == user_id]
            high_rated_movies = user_ratings[user_ratings['rating'] >= 4.0]['movieId'].values
            
            if len(high_rated_movies) > 0:
                try:
                    recommendations = self.knn_recommend_for_user(user_id, 10)
                    if not recommendations.empty:
                        recommended_movies = recommendations['movieId'].values
                        
                        # Check if any recommended movies were actually highly rated
                        for movie_id in high_rated_movies:
                            if movie_id in recommended_movies:
                                hits += 1
                                break
                        total += 1
                except:
                    continue
        
        accuracy = hits / total if total > 0 else 0
        print(f"KNN Hit Rate: {accuracy:.3f}")
        return accuracy

    # Your original methods (maintained for app.py compatibility)
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

    # New KNN + RF methods
    def knn_recommend_for_user(self, user_id, top_n=10):
        """KNN-based recommendations for user"""
        if user_id not in self.user_movie_matrix.index:
            return pd.DataFrame()
        
        user_ratings = self.user_movie_matrix.loc[user_id].values.reshape(1, -1)
        distances, indices = self.knn.kneighbors(user_ratings)
        
        similar_users = self.user_movie_matrix.iloc[indices[0]]
        recommendations = similar_users.mean().sort_values(ascending=False)
        
        top_movie_ids = recommendations.head(top_n).index
        return self.movies_with_stats[
            self.movies_with_stats['movieId'].isin(top_movie_ids)
        ][['movieId', 'title', 'genres', 'rating_mean', 'rating_count']]
    
    def hybrid_personal_recommendations(self, user_id, top_n=10):
        """Hybrid recommendations combining KNN and Random Forest"""
        # Get KNN recommendations
        knn_recs = self.knn_recommend_for_user(user_id, top_n * 3)
        
        if knn_recs.empty:
            return self.get_popular_movies(top_n)
        
        # Score each recommendation (simplified for VS Code)
        scored_movies = []
        for _, movie in knn_recs.iterrows():
            final_score = (
                0.7 * movie['rating_mean'] +  # Historical rating
                0.3 * min(1.0, movie['rating_count'] / 100)  # Popularity boost
            )
            
            scored_movies.append({
                'movieId': movie['movieId'],
                'title': movie['title'],
                'genres': movie['genres'],
                'rating_mean': movie['rating_mean'],
                'rating_count': movie['rating_count'],
                'final_score': final_score
            })
        
        # Sort by final score
        scored_df = pd.DataFrame(scored_movies)
        return scored_df.sort_values('final_score', ascending=False).head(top_n)

def train_and_save_models():
    """Train models and save them as PKL files"""
    print("Loading datasets...")
    
    try:
        # Load data - flexible for different file combinations
        movies = pd.read_csv('data/movies.csv')
        ratings = pd.read_csv('data/ratings.csv')
        
        # Try to load optional files
        try:
            tags = pd.read_csv('data/tags.csv')
            print("Tags file loaded successfully!")
        except:
            tags = None
            print("No tags file found, continuing without tags...")
        
        try:
            links = pd.read_csv('data/links.csv')
            print("Links file loaded successfully!")
        except:
            links = None
            print("No links file found, continuing without links...")
        
        print(f"Movies: {movies.shape}")
        print(f"Ratings: {ratings.shape}")
        if tags is not None:
            print(f"Tags: {tags.shape}")
        
    except Exception as e:
        print(f"Error loading files: {e}")
        print("Please make sure movies.csv and ratings.csv exist in the data folder")
        return None
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Train recommender
    print("Training enhanced recommender with KNN + Random Forest...")
    recommender = EnhancedMovieRecommender(movies, ratings, tags)
    
    # Evaluate models
    print("\nModel Evaluation:")
    knn_accuracy = recommender.evaluate_knn()
    
    # Test the model
    print("\nTesting model...")
    print("Popular movies sample:")
    popular = recommender.get_popular_movies(3)
    for _, movie in popular.iterrows():
        print(f"  - {movie['title']} (Rating: {movie['rating_mean']})")
    
    # Test KNN recommendations if we have enough users
    if len(ratings['userId'].unique()) > 1:
        sample_user = ratings['userId'].iloc[0]
        print(f"\nKNN Recommendations for User {sample_user}:")
        knn_recs = recommender.knn_recommend_for_user(sample_user, 3)
        for _, movie in knn_recs.iterrows():
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
    
    # Save KNN and RF models
    with open('models/knn_model.pkl', 'wb') as f:
        pickle.dump(recommender.knn, f)
    
    with open('models/random_forest_model.pkl', 'wb') as f:
        pickle.dump(recommender.rf_model, f)
    
    print("✅ All models saved successfully!")
    print("📁 Files created in 'models/' folder:")
    model_files = os.listdir('models')
    for file in model_files:
        print(f"   - {file}")
    
    print(f"\n🎯 Final Model Performance:")
    print(f"   KNN Hit Rate: {knn_accuracy:.3f}")
    
    return recommender

if __name__ == "__main__":
    print("=" * 60)
    print("MOVIE RECOMMENDER - KNN + RANDOM FOREST HYBRID")
    print("=" * 60)
    
    recommender = train_and_save_models()
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED!")
    print("=" * 60)
    print("\nNext: Run 'app.py' to start the Streamlit app!")