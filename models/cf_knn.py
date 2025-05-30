from utils.dataset import get_user_history
import random

def recommend(user_id=None, liked_movies=None, top_n=5):
    # Ưu tiên dùng lịch sử nếu không truyền liked_movies
    if not liked_movies and user_id:
        liked_movies = get_user_history(user_id)

    if not liked_movies:
        return ["Không tìm thấy phim phù hợp"]

    # Tạo danh sách gợi ý ngẫu nhiên từ tên phim đã xem
    pool = [title for title in liked_movies if isinstance(title, str)]
    pool = list(set(pool))  # loại trùng

    random.shuffle(pool)
    return pool[:top_n] if pool else ["Không tìm thấy phim phù hợp"]