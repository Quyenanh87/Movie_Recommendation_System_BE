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

# import pickle
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np

# # Load mô hình đã lưu
# with open('tfidf_recommender.pkl', 'rb') as f:
#     model = pickle.load(f)

# vec = model['vectorizer']
# movieId_to_content = model['movieId_to_content']
# movies = model['movies_df']
# user_train_rating = model['user_train_rating']

# # Tạo profile người dùng
# def build_user_profile_tfidf(user, user_train_dict, movieId_to_content, vec):
#     movie_ids = user_train_dict.get(user, [])
#     docs = [movieId_to_content[mid] for mid in movie_ids if mid in movieId_to_content]
#     if not docs:
#         return None
#     X = vec.transform(docs)
#     return np.mean(X.toarray(), axis=0)

# # Hàm gợi ý chỉ cần user_id
# def recommend_topk_for_user(user_id, top_k=10):
#     if user_id not in user_train_rating:
#         print(f"❌ User {user_id} không tồn tại trong dữ liệu huấn luyện.")
#         return

#     profile = build_user_profile_tfidf(user_id, user_train_rating, movieId_to_content, vec)
#     if profile is None:
#         print("❌ Không thể xây dựng hồ sơ người dùng.")
#         return

#     # Loại bỏ các phim user đã xem
#     train_set = set(user_train_rating[user_id])
#     candidates = list(set(movieId_to_content.keys()) - train_set)

#     sims = []
#     for mid in candidates:
#         vec_movie = vec.transform([movieId_to_content[mid]]).toarray()[0]
#         sim = cosine_similarity(profile.reshape(1, -1), vec_movie.reshape(1, -1))[0][0]
#         sims.append((mid, sim))

#     # Sắp xếp theo độ tương đồng giảm dần
#     top_items = sorted(sims, key=lambda x: -x[1])[:top_k]

#     print(f"\n🎯 Gợi ý Top-{top_k} phim cho user {user_id}:")
#     for mid, sim in top_items:
#         title = movies.loc[movies['movieId'] == mid, 'title'].values[0]
#         print(f"{title} (similarity: {sim:.4f})")

