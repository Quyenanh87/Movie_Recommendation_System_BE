import pickle
import numpy as np
import pandas as pd
from utils.dataset import get_user_history

print("[DEBUG] Đang load ItemCF model...")

# Load model từ file pkl
with open("file_pkl/itemcf_model.pkl", "rb") as f:
    model = pickle.load(f)

# Extract các thành phần cần thiết
sim_df = model["sim_df"]  # Ma trận similarity giữa các item
train_rating_dict = model["train_rating_dict"]  # Dictionary chứa rating train
item_avg_rating = model["item_avg_rating"]  # Rating trung bình của mỗi item
user_train_rating = model["user_train_rating"]  # Dictionary user -> list phim đã xem
movies_df = model["movies_df"]  # DataFrame chứa thông tin phim

print("[DEBUG] Đã load xong ItemCF model")

def predict_rating_item_item(user, movie, user_train_dict, sim_df, train_rating_dict, item_avg_rating, neighbor=10):
    """Dự đoán rating cho (user, movie) dựa trên Item-CF"""
    rated_movies = user_train_dict.get(user, [])
    if len(rated_movies) == 0 or movie not in sim_df.index:
        return np.nan

    sim_list = []
    for rated_movie in rated_movies:
        if rated_movie in sim_df.index:
            sim = sim_df.at[movie, rated_movie]
            r_val = train_rating_dict.get((user, rated_movie))
            if r_val is not None and not np.isnan(sim):
                diff = r_val - item_avg_rating.get(rated_movie, 0)
                sim_list.append((sim, diff))

    if len(sim_list) == 0:
        user_ratings = [train_rating_dict[(user, m)] for m in rated_movies if (user, m) in train_rating_dict]
        return np.mean(user_ratings) if len(user_ratings) > 0 else np.nan

    sim_list = sorted(sim_list, key=lambda x: abs(x[0]), reverse=True)[:neighbor]
    similarities = np.array([s for s, _ in sim_list])
    rating_diffs = np.array([d for _, d in sim_list])
    weighted_sum = np.dot(similarities, rating_diffs)
    sum_sim = np.sum(np.abs(similarities))
    pred_diff = weighted_sum / sum_sim if sum_sim != 0 else 0
    pred = item_avg_rating.get(movie, 0) + pred_diff
    return np.clip(pred, 1, 5)

def recommend_for_user(user_id, top_k=10):
    """Hàm gợi ý top-k phim cho một user dựa trên ItemCF"""
    if user_id not in user_train_rating:
        print(f"❌ User {user_id} không có trong dữ liệu huấn luyện.")
        return {
            "success": False,
            "message": f"User {user_id} không có trong dữ liệu huấn luyện.",
            "recommendations": []
        }

    # Lấy tất cả phim chưa xem làm candidates
    watched = set(user_train_rating[user_id])
    all_movies = set(item_avg_rating.keys())
    candidates = list(all_movies - watched)
    
    if not candidates:
        print(f"⚠️ Không tìm thấy phim phù hợp cho user {user_id}.")
        return {
            "success": False,
            "message": "Không tìm thấy phim phù hợp để gợi ý.",
            "recommendations": []
        }

    # Tính điểm cho các phim candidate
    predictions = []
    for movie in candidates:
        pred = predict_rating_item_item(
            user_id, movie,
            user_train_rating,
            sim_df,
            train_rating_dict,
            item_avg_rating,
            neighbor=10
        )
        if not np.isnan(pred):
            predictions.append((movie, pred))

    # Sắp xếp và lấy top-k phim
    top_items = sorted(predictions, key=lambda x: -x[1])[:top_k]
    
    print(f"\n🎯 Top-{top_k} phim gợi ý cho user {user_id}:")
    
    recommendations = []
    for movie_id, score in top_items:
        title = movies_df.loc[movies_df['movieId'] == movie_id, 'title'].values[0]
        print(f"{title} — predicted rating: {score:.2f}")
        recommendations.append({
            "title": title,
            "predicted_rating": float(score),
            "movieId": int(movie_id)
        })

    return {
        "success": True,
        "message": f"Đã tìm thấy {len(recommendations)} gợi ý phù hợp",
        "recommendations": recommendations
    }

def recommend(user_id=None, liked_movies=None, top_n=5):
    """Hàm recommend chính để tương thích với API"""
    # Ưu tiên dùng lịch sử nếu không truyền liked_movies
    if not liked_movies and user_id:
        liked_movies = get_user_history(user_id)
        print("[DEBUG] Lịch sử phim đã xem:", liked_movies)

    if not liked_movies:
        print("[DEBUG] Không tìm thấy lịch sử xem phim")
        return ["Không tìm thấy phim phù hợp"]

    # Nếu có user_id, dùng ItemCF để gợi ý
    if user_id:
        print(f"\n[DEBUG] Đang tìm gợi ý cho user {user_id}...")
        result = recommend_for_user(user_id, top_k=top_n)
        if result["success"]:
            recommendations = [rec["title"] for rec in result["recommendations"]]
            print(f"[DEBUG] Danh sách phim gợi ý: {recommendations}")
            return recommendations
        print("[DEBUG] Không tìm được gợi ý phù hợp")
        return ["Không tìm thấy phim phù hợp"]

    # Nếu không có user_id nhưng có liked_movies, trả về top_n phim từ danh sách
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