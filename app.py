# app.py - COMPLETE VERSION WITH CLASS DEFINITION
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Define the EnhancedMovieRecommender class here
class EnhancedMovieRecommender:
    def __init__(self, movies_df, ratings_df):
        self.movies = movies_df
        self.ratings = ratings_df
        self._prepare_data()
        self._build_similarity_matrix()
    
    def _prepare_data(self):
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
    
    def _build_similarity_matrix(self):
        self.tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = self.tfidf.fit_transform(self.movies_with_stats['genres'])
        self.cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
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

# Configure page
st.set_page_config(
    page_title="What to Watch Tonight",
    page_icon="🎬",
    layout="wide"
)

def load_models():
    """Load models with the class definition available"""
    try:
        # Check if models exist
        if not os.path.exists('models/full_recommender.pkl'):
            return None
        
        # Load the recommender
        with open('models/full_recommender.pkl', 'rb') as f:
            recommender = pickle.load(f)
        
        return recommender
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

def main():
    st.title("🎬 What to Watch Tonight")
    st.markdown("### Find your perfect movie for tonight!")
    
    # Load models
    with st.spinner("Loading recommendation models..."):
        recommender = load_models()
    
    if recommender is None:
        st.error("❌ Models not loaded. Please make sure:")
        st.info("1. You've run `python train_models.py` successfully")
        st.info("2. The 'models' folder exists with PKL files")
        
        # Show current directory contents
        st.write("**Current folder contents:**")
        current_files = os.listdir('.')
        st.write(f"Files: {[f for f in current_files if f.endswith('.py') or f.endswith('.csv')]}")
        
        if os.path.exists('models'):
            st.write("**Models folder contents:**")
            model_files = os.listdir('models')
            st.write(f"Model files: {model_files}")
        else:
            st.write("❌ 'models' folder not found!")
        
        return
    
    st.success("✅ Models loaded successfully!")
    
    # Sidebar filters
    st.sidebar.header("🎯 Your Preferences")
    
    # Recommendation type
    rec_type = st.sidebar.radio(
        "Choose recommendation type:",
        ["Popular Movies", "Mood-Based", "Similar Movies", "Custom Filter"]
    )
    
    if rec_type == "Popular Movies":
        st.subheader("🎉 Most Popular Movies")
        st.write("These are the highest rated movies with the most votes:")
        
        num_movies = st.slider("Number of movies to show", 5, 20, 10)
        min_votes = st.slider("Minimum number of votes", 10, 200, 50)
        
        popular_movies = recommender.get_popular_movies(num_movies, min_ratings=min_votes)
        
        if len(popular_movies) > 0:
            for idx, (_, movie) in enumerate(popular_movies.iterrows(), 1):
                with st.container():
                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1:
                        st.write(f"**{idx}. {movie['title']}**")
                        st.write(f"*{movie['genres']}*")
                    with col2:
                        st.metric("Rating", f"{movie['rating_mean']:.1f}/5")
                    with col3:
                        st.metric("Votes", f"{int(movie['rating_count'])}")
                    st.divider()
        else:
            st.warning("No popular movies found with the current filters.")
    
    elif rec_type == "Mood-Based":
        st.subheader("😊 Mood-Based Recommendations")
        st.write("Tell us how you're feeling and we'll suggest the perfect movies!")
        
        mood = st.selectbox(
            "How are you feeling?",
            ["funny", "romantic", "action", "drama", "scary", "mind_bending", "family", "inspirational"],
            format_func=lambda x: {
                "funny": "😂 Funny & Lighthearted",
                "romantic": "💕 Romantic & Heartwarming", 
                "action": "💥 Action & Adventure",
                "drama": "🎭 Drama & Emotional",
                "scary": "👻 Scary & Thrilling",
                "mind_bending": "🧠 Mind-Bending & Thoughtful",
                "family": "👨‍👩‍👧‍👦 Family & Kids Friendly",
                "inspirational": "🌟 Inspirational & Motivational"
            }[x]
        )
        
        num_movies = st.slider("Number of movies", 5, 15, 8)
        
        if st.button("Get Mood Recommendations", type="primary"):
            with st.spinner(f"Finding perfect {mood} movies..."):
                mood_movies = recommender.get_mood_recommendations(mood, num_movies)
            
            if len(mood_movies) > 0:
                st.success(f"Here are some great {mood} movies:")
                for idx, (_, movie) in enumerate(mood_movies.iterrows(), 1):
                    with st.expander(f"{idx}. {movie['title']}"):
                        st.write(f"**Genres:** {movie['genres']}")
                        st.write(f"**Rating:** {movie['rating_mean']:.1f}/5 ({int(movie['rating_count'])} votes)")
                        st.write(f"**Year:** {movie['year']}")
            else:
                st.warning(f"No {mood} movies found. Try a different mood!")
    
    elif rec_type == "Similar Movies":
        st.subheader("🔍 Find Similar Movies")
        st.write("Discover movies similar to ones you already love!")
        
        # Get a sample of popular movies for selection
        popular_titles = recommender.get_popular_movies(50)['title'].tolist()
        movie_title = st.selectbox(
            "Choose a movie you like:",
            sorted(popular_titles)
        )
        
        num_similar = st.slider("Number of similar movies", 3, 10, 6)
        
        if st.button("Find Similar Movies", type="primary"):
            with st.spinner(f"Finding movies similar to '{movie_title}'..."):
                similar_movies = recommender.get_similar_movies(movie_title, num_similar)
            
            if similar_movies is not None and len(similar_movies) > 0:
                st.success(f"If you liked **{movie_title}**, you might also enjoy:")
                for idx, (_, movie) in enumerate(similar_movies.iterrows(), 1):
                    with st.container():
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            st.write(f"**{idx}. {movie['title']}**")
                            st.write(f"*{movie['genres']}*")
                        with col2:
                            st.metric("Rating", f"{movie['rating_mean']:.1f}/5")
                        st.divider()
            else:
                st.error("Movie not found in database or no similar movies found.")
    
    elif rec_type == "Custom Filter":
        st.subheader("🎛️ Custom Recommendations")
        st.write("Fine-tune your movie search with specific filters!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            selected_genres = st.multiselect(
                "Preferred genres:",
                ["Action", "Comedy", "Drama", "Romance", "Thriller", "Horror", "Sci-Fi", "Animation", "Adventure", "Crime"]
            )
            min_rating = st.slider("Minimum rating", 1.0, 5.0, 3.0, 0.5)
        
        with col2:
            min_votes = st.slider("Minimum votes", 0, 100, 10)
            num_results = st.slider("Number of results", 5, 20, 10)
        
        if st.button("Get Custom Recommendations", type="primary"):
            with st.spinner("Finding your perfect matches..."):
                custom_recs = recommender.hybrid_recommendations(
                    genres=selected_genres,
                    min_rating=min_rating,
                    min_ratings=min_votes,
                    top_n=num_results
                )
            
            if len(custom_recs) > 0:
                st.success(f"Found {len(custom_recs)} movies matching your criteria:")
                for idx, (_, movie) in enumerate(custom_recs.iterrows(), 1):
                    with st.expander(f"🎭 {idx}. {movie['title']}"):
                        st.write(f"**Genres:** {movie['genres']}")
                        st.write(f"**Rating:** {movie['rating_mean']:.1f}/5 ({int(movie['rating_count'])} votes)")
                        st.write(f"**Year:** {movie['year']}")
                        st.write(f"**Weighted Score:** {movie['weighted_score']:.2f}")
            else:
                st.warning("No movies match your criteria. Try relaxing your filters!")

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info(
        "💡 **Tip:** Try different recommendation types to discover new movies!"
    )

if __name__ == "__main__":
    main()