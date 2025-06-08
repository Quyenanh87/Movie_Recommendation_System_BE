import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from utils.dataset import get_user_history, find_matching_titles

# Load mô hình
with open('file_pkl/tfidf_recommender.pkl', 'rb') as f:
    model = pickle.load(f)

vec = model['vectorizer']
movieId_to_content = model['movieId_to_content']
movies = model['movies_df']
user_train_rating = model['user_train_rating']

# Hàm tạo profile
def build_user_profile_tfidf(user, user_train_dict, movieId_to_content, vec):
    movie_ids = user_train_dict.get(user, [])
    docs = [movieId_to_content[mid] for mid in movie_ids if mid in movieId_to_content]
    if not docs:
        return None
    X = vec.transform(docs)
    return np.mean(X.toarray(), axis=0)

# Gợi ý Top-K phim
def recommend_topk_for_user(user_id, top_k=10):
    if user_id not in user_train_rating:
        return {
            "success": False,
            "message": f"User {user_id} không tồn tại trong dữ liệu huấn luyện.",
            "recommendations": []
        }

    profile = build_user_profile_tfidf(user_id, user_train_rating, movieId_to_content, vec)
    if profile is None:
        return {
            "success": False,
            "message": "Không thể xây dựng hồ sơ người dùng.",
            "recommendations": []
        }

    train_set = set(user_train_rating[user_id])
    candidates = list(set(movieId_to_content.keys()) - train_set)

    sims = []
    for mid in candidates:
        vec_movie = vec.transform([movieId_to_content[mid]]).toarray()[0]
        sim = cosine_similarity(profile.reshape(1, -1), vec_movie.reshape(1, -1))[0][0]
        sims.append((mid, sim))

    top_items = sorted(sims, key=lambda x: -x[1])[:top_k]
    
    recommendations = []
    for mid, sim in top_items:
        title = movies.loc[movies['movieId'] == mid, 'title'].values[0]
        recommendations.append({
            "title": title,
            "similarity": float(sim),
            "movieId": int(mid)
        })

    print("\n[DEBUG] Chi tiết gợi ý:")
    for rec in recommendations:
        print(f"- {rec['title']} (Độ tương đồng: {rec['similarity']:.4f})")

    return {
        "success": True,
        "message": f"Đã tìm thấy {len(recommendations)} gợi ý phù hợp",
        "recommendations": recommendations
    }

def recommend(user_id=None, liked_movies=None, top_n=5):
    """
    Hàm recommend chính để tương thích với API
    """
    # Ưu tiên dùng lịch sử nếu không truyền liked_movies
    if not liked_movies and user_id:
        liked_movies = get_user_history(user_id)
        print("[DEBUG] Lịch sử phim đã xem:", liked_movies)

    if not liked_movies:
        print("[DEBUG] Không tìm thấy lịch sử xem phim")
        return ["Không tìm thấy phim phù hợp"]

    # Nếu có user_id, dùng collaborative filtering
    if user_id:
        print(f"\n[DEBUG] Đang tìm gợi ý cho user {user_id}...")
        result = recommend_topk_for_user(user_id, top_k=top_n)
        if result["success"]:
            recommendations = [rec["title"] for rec in result["recommendations"]]
            print(f"[DEBUG] Danh sách phim gợi ý: {recommendations}")
            return recommendations
        print("[DEBUG] Không tìm được gợi ý phù hợp")
        return ["Không tìm thấy phim phù hợp"]

    # Nếu không có user_id nhưng có liked_movies, trả về random từ liked_movies
    print("\n[DEBUG] Đang xử lý danh sách phim yêu thích...")
    pool = [title for title in liked_movies if isinstance(title, str)]
    pool = list(set(pool))  # loại trùng
    print(f"[DEBUG] Danh sách phim hợp lệ: {pool}")
    
    if not pool:
        print("[DEBUG] Không có phim hợp lệ trong danh sách")
        return ["Không tìm thấy phim phù hợp"]
    
    result = pool[:top_n]
    print(f"[DEBUG] Kết quả gợi ý cuối cùng: {result}")
    return result