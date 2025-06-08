import pickle
import numpy as np
import pandas as pd
from utils.dataset import get_user_history

print("[DEBUG] Đang load UserCF model...")

# Load model từ file pkl
with open("file_pkl/usercf_model.pkl", "rb") as f:
    model = pickle.load(f)

# Extract các thành phần cần thiết
sim_df = model["sim_df"]  # Ma trận similarity giữa các user
train_rating_dict = model["train_rating_dict"]  # Dictionary chứa rating train
user_avg_rating = model["user_avg_rating"]  # Rating trung bình của mỗi user
user_train_rating = model["user_train_rating"]  # Dictionary user -> list phim đã xem
movies_df = model["movies_df"]  # DataFrame chứa thông tin phim

print("[DEBUG] Đã load xong UserCF model")

def get_top_similar_users(user_id, sim_df, n_similar=50):
    """Lấy top N users tương đồng nhất với user_id"""
    if user_id not in sim_df.index:
        return []
    user_similarities = sim_df.loc[user_id]
    # Loại bỏ chính user_id và sắp xếp giảm dần
    similar_users = user_similarities.drop(user_id).sort_values(ascending=False)
    return similar_users[:n_similar]

def get_candidate_items(user_id, similar_users, user_train_rating, train_rating_dict, top_n=100):
    """Lấy danh sách phim candidate từ các users tương đồng"""
    watched_movies = set(user_train_rating.get(user_id, []))
    candidate_items = {}
    
    for sim_user, sim_score in similar_users.items():
        # Lấy các phim của user tương đồng
        for (u, m), rating in train_rating_dict.items():
            if u == sim_user and m not in watched_movies:
                if m not in candidate_items:
                    candidate_items[m] = {'count': 0, 'total_sim': 0}
                candidate_items[m]['count'] += 1
                candidate_items[m]['total_sim'] += sim_score
    
    # Sắp xếp theo độ phổ biến (weighted by similarity)
    sorted_items = sorted(
        candidate_items.items(),
        key=lambda x: (x[1]['count'] * x[1]['total_sim']),
        reverse=True
    )
    
    return [item[0] for item in sorted_items[:top_n]]

def predict_rating_user_user(user, movie, user_train_dict, sim_df, train_rating_dict, user_avg_rating, neighbor=10):
    """Dự đoán rating cho (user, movie) dựa trên User-CF"""
    users_who_rated = [v for (v, m) in train_rating_dict.keys() if m == movie and v != user]
    
    if len(users_who_rated) == 0 or user not in sim_df.index:
        return user_avg_rating.get(user, np.nan)
    
    sim_list = []
    for v in users_who_rated:
        if v in sim_df.columns:
            sim = sim_df.at[user, v]
            r_val = train_rating_dict.get((v, movie))
            if r_val is not None and not np.isnan(sim):
                diff = r_val - user_avg_rating.get(v, 0)
                sim_list.append((sim, diff))
    
    if len(sim_list) == 0:
        return user_avg_rating.get(user, np.nan)
    
    sim_list = sorted(sim_list, key=lambda x: abs(x[0]), reverse=True)[:neighbor]
    similarities = np.array([s for s, _ in sim_list])
    rating_diffs = np.array([d for _, d in sim_list])
    
    weighted_sum = np.dot(similarities, rating_diffs)
    sum_sim = np.sum(np.abs(similarities))
    pred_diff = weighted_sum / sum_sim if sum_sim != 0 else 0
    pred = user_avg_rating.get(user, 0) + pred_diff
    return np.clip(pred, 1, 5)

def recommend_for_user(user_id, top_k=10):
    """Hàm gợi ý top-k phim cho một user dựa trên UserCF"""
    if user_id not in user_train_rating:
        print(f"❌ User {user_id} không có trong dữ liệu huấn luyện.")
        return {
            "success": False,
            "message": f"User {user_id} không có trong dữ liệu huấn luyện.",
            "recommendations": []
        }

    # Lấy top users tương đồng
    similar_users = get_top_similar_users(user_id, sim_df)
    if len(similar_users) == 0:
        print(f"⚠️ Không tìm thấy users tương đồng với user {user_id}.")
        return {
            "success": False,
            "message": "Không tìm thấy users tương đồng.",
            "recommendations": []
        }

    # Lấy candidate items từ users tương đồng
    candidate_movies = get_candidate_items(
        user_id, similar_users, 
        user_train_rating, train_rating_dict
    )
    
    if not candidate_movies:
        print(f"⚠️ Không tìm thấy phim phù hợp cho user {user_id}.")
        return {
            "success": False,
            "message": "Không tìm thấy phim phù hợp để gợi ý.",
            "recommendations": []
        }

    # Tính điểm cho các phim candidate
    predictions = []
    for movie in candidate_movies:
        pred = predict_rating_user_user(
            user_id, movie,
            user_train_rating,
            sim_df,
            train_rating_dict,
            user_avg_rating,
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
        print(f"{title} (predicted rating: {score:.2f})")
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

    # Nếu có user_id, dùng UserCF để gợi ý
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