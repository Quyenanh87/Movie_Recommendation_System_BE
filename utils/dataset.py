import pandas as pd

# Đọc file CSV thay vì .dat
df_movies = pd.read_csv("data/movies.csv", encoding="utf-8")  
df_ratings = pd.read_csv("data/ratings.csv", encoding="utf-8")  

# Tạo cột 'content' để dùng cho TF-IDF, BERT...
df_movies["content"] = df_movies["title"] + " " + df_movies["genres"].fillna("")

def get_movies_dataframe():
    return df_movies

def get_ratings_dataframe():
    return df_ratings

def get_available_titles(limit=100):
    return df_movies["title"].dropna().sample(n=limit, random_state=42).tolist()


def get_user_history(user_id: int, limit: int = 10) -> list[str]:
    """
    Trả về danh sách tên phim user đã xem gần đây (mặc định lấy 10 phim mới nhất).
    """
    ratings = get_ratings_dataframe()
    movies = get_movies_dataframe()

    # Lọc các rating của user
    user_ratings = ratings[ratings["userId"] == user_id]

    if user_ratings.empty:
        return []

    # Sắp xếp theo thời gian và nối với bảng movie
    recent = user_ratings.sort_values("timestamp", ascending=False).head(limit)
    merged = recent.merge(movies, on="movieId")

    return merged["title"].dropna().tolist()
def find_matching_titles(user_input_titles, all_titles):
    matched_titles = []
    for input_title in user_input_titles:
        for real_title in all_titles:
            if input_title.strip().lower() == real_title.strip().lower():
                matched_titles.append(real_title)
                break
    return matched_titles
