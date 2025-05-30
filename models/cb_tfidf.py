import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils.dataset import get_movies_dataframe, get_user_history, find_matching_titles

# Load dữ liệu
df = get_movies_dataframe()
df["content"] = df["title"] + " " + df["genres"].fillna("")

# TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df["content"])

# Mapping từ title → index
movie_indices = pd.Series(df.index, index=df["title"]).drop_duplicates()

# Gợi ý
def recommend(user_id=None, liked_movies=None, top_n=5):
    if not liked_movies and user_id:
        liked_movies = get_user_history(user_id)
        print("[DEBUG] Liked from history:", liked_movies)

    matched_titles = find_matching_titles(liked_movies or [], movie_indices.index.tolist())
    print("[DEBUG] Matched titles:", matched_titles)

    indices = [movie_indices.get(m) for m in matched_titles if movie_indices.get(m) is not None]

    if not indices:
        return ["Không tìm thấy phim phù hợp"]

    user_vector = tfidf_matrix[indices].mean(axis=0)
    user_vector = np.asarray(user_vector)

    similarity_scores = cosine_similarity(user_vector, tfidf_matrix).flatten()
    similar_indices = similarity_scores.argsort()[::-1]

    recommended = []
    for idx in similar_indices:
        movie = df.iloc[idx]["title"]
        if movie not in matched_titles and movie not in recommended:
            recommended.append(movie)
        if len(recommended) >= top_n:
            break

    print("[🎯] Gợi ý:", recommended)
    return recommended
