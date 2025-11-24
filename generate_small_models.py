# generate_small_models.py
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

print("Generating small PKL files for deployment...")

# Load only small subset of data
movies = pd.read_csv('data/movies.csv').head(500)  # Only 500 movies
ratings = pd.read_csv('data/ratings.csv')
ratings = ratings[ratings['movieId'].isin(movies['movieId'])].head(10000)  # Only 10k ratings

print(f"Using: {len(movies)} movies, {len(ratings)} ratings")

# Create models directory
os.makedirs('models', exist_ok=True)

# 1. Create small similarity matrix
tfidf = TfidfVectorizer(stop_words='english', max_features=100)  # Only 100 features
tfidf_matrix = tfidf.fit_transform(movies['genres'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix).astype(np.float16)  # Half precision

with open('models/cosine_similarity.pkl', 'wb') as f:
    pickle.dump(cosine_sim, f, protocol=4)
print("✓ cosine_similarity.pkl")

# 2. Save TF-IDF vectorizer
with open('models/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f, protocol=4)
print("✓ tfidf_vectorizer.pkl")

# 3. Create and save movies with stats
movie_stats = ratings.groupby('movieId').agg({'rating': ['count', 'mean']}).round(3)
movie_stats.columns = ['rating_count', 'rating_mean']
movies_with_stats = movies.merge(movie_stats, on='movieId', how='left')
movies_with_stats['rating_count'] = movies_with_stats['rating_count'].fillna(0)
movies_with_stats['rating_mean'] = movies_with_stats['rating_mean'].fillna(0)
movies_with_stats['weighted_score'] = movies_with_stats['rating_mean'] * np.log1p(movies_with_stats['rating_count'])

with open('models/movies_with_stats.pkl', 'wb') as f:
    pickle.dump(movies_with_stats, f, protocol=4)
print("✓ movies_with_stats.pkl")

# 4. Create small KNN model
user_movie_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
knn = NearestNeighbors(n_neighbors=10, metric='cosine')
knn.fit(user_movie_matrix)

with open('models/knn_model.pkl', 'wb') as f:
    pickle.dump(knn, f, protocol=4)
print("✓ knn_model.pkl")

# 5. Create dummy files for the rest (your app.py might need them)
class DummyModel:
    pass

dummy = DummyModel()

with open('models/full_recommender.pkl', 'wb') as f:
    pickle.dump(dummy, f, protocol=4)
print("✓ full_recommender.pkl")

with open('models/random_forest_model.pkl', 'wb') as f:
    pickle.dump(dummy, f, protocol=4)
print("✓ random_forest_model.pkl")

# Check file sizes
print("\nFile sizes:")
total_size = 0
for file in os.listdir('models'):
    size = os.path.getsize(f'models/{file}') / (1024*1024)
    total_size += size
    print(f"  {file}: {size:.2f} MB")

print(f"\n✅ Total size: {total_size:.2f} MB - Ready for GitHub!")
print("Now run: git add . && git commit -m 'Add small models' && git push")