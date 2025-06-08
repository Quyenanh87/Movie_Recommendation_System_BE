import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from utils.dataset import get_user_history, find_matching_titles

print("[DEBUG] Đang load model từ file pkl...")
# Load mô hình
with open('file_pkl/ridge_tfidf_recommender.pkl', 'rb') as f:
    model = pickle.load(f)

print("[DEBUG] Các thành phần trong model:", list(model.keys()))

# Trích xuất các thành phần từ model
vec = model['vectorizer']
movieId_to_content = model['movieId_to_content']
movies = model['movies_df']
user_models = model['user_models']
user_train_rating = model['user_train_rating']

def recommend_topk_for_user(user_id, top_k=10):
    """
    Hàm gợi ý top-k phim cho một user sử dụng Ridge model
    """
    if user_id not in user_models:
        print(f"❌ User {user_id} không có mô hình được huấn luyện.")
        return {
            "success": False,
            "message": f"User {user_id} không có mô hình được huấn luyện.",
            "recommendations": []
        }

    # Lấy model của user
    model = user_models[user_id]
    
    # Lấy danh sách phim đã xem để loại trừ
    train_set = set(user_train_rating.get(user_id, []))
    # Lấy tất cả movie_ids có trong movieId_to_content
    all_movie_ids = set(movieId_to_content.keys())
    # Lọc ra các phim chưa xem
    candidate_ids = list(all_movie_ids - train_set)

    # Dự đoán điểm cho từng phim
    preds = {}
    raw_scores = []
    movie_ids = []
    
    for mid in candidate_ids:
        content = movieId_to_content.get(mid, "")
        if not content:
            continue
        X = vec.transform([content]).toarray()
        score = model.predict(X)[0]
        raw_scores.append(score)
        movie_ids.append(mid)

    if not raw_scores:
        return {
            "success": False,
            "message": "Không tìm được phim phù hợp để gợi ý.",
            "recommendations": []
        }

    # Chuẩn hóa điểm về khoảng [1,5] sử dụng MinMaxScaler
    scaler = MinMaxScaler(feature_range=(1, 5))
    normalized_scores = scaler.fit_transform(np.array(raw_scores).reshape(-1, 1)).flatten()

    # Tạo dictionary với điểm đã chuẩn hóa
    for mid, raw_score, norm_score in zip(movie_ids, raw_scores, normalized_scores):
        preds[mid] = {
            'raw': raw_score,
            'normalized': norm_score
        }

    # Sắp xếp theo điểm chuẩn hóa giảm dần, nếu bằng nhau thì theo movieId
    top_items = sorted(preds.items(), key=lambda x: (-x[1]['normalized'], x[0]))[:top_k]
    
    print(f"\n🎯 Gợi ý Top-{top_k} phim cho user {user_id}:")
    print(f"[DEBUG] Thống kê điểm dự đoán gốc:")
    print(f"Min raw: {min(raw_scores):.4f}")
    print(f"Max raw: {max(raw_scores):.4f}")
    print(f"Mean raw: {np.mean(raw_scores):.4f}")
    print(f"Std raw: {np.std(raw_scores):.4f}")
    
    recommendations = []
    for mid, scores in top_items:
        title = movies.loc[movies['movieId'] == mid, 'title'].values[0]
        print(f"{title} (normalized: {scores['normalized']:.4f}, raw: {scores['raw']:.4f})")
        recommendations.append({
            "title": title,
            "similarity": float(scores['normalized']),
            "movieId": int(mid)
        })

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

    if not liked_movies:
        return ["Không tìm thấy phim phù hợp"]

    # Nếu có user_id, dùng Ridge model để gợi ý
    if user_id:
        result = recommend_topk_for_user(user_id, top_k=top_n)
        if result["success"]:
            recommendations = [rec["title"] for rec in result["recommendations"]]
            return recommendations
        return ["Không tìm thấy phim phù hợp"]

    # Nếu không có user_id nhưng có liked_movies, trả về top_n phim từ danh sách
    pool = [title for title in liked_movies if isinstance(title, str)]
    pool = list(set(pool))  # loại trùng
    
    if not pool:
        return ["Không tìm thấy phim phù hợp"]
    
    result = pool[:top_n]
    return result