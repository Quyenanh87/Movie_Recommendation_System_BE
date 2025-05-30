import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from utils.dataset import get_movies_dataframe, get_user_history, find_matching_titles

# Load mô hình và vectorizer từ file_pkl
with open("file_pkl/user_models_cb_tfidf_ridge.pkl", "rb") as f:
    user_models = pickle.load(f)

with open("file_pkl/tfidf_vectorizer_cb_tfidf_ridge.pkl", "rb") as f:
    tfidf_vectorizer = pickle.load(f)

# Load dữ liệu phim
movies = get_movies_dataframe()
movies['content'] = (movies['title'].fillna('') + ' ' + movies['genres'].fillna('')).str.replace('|', ' ').str.lower()
movieId_to_content = dict(zip(movies['movieId'], movies['content']))

def get_movie_vector(mid):
    content = movieId_to_content.get(mid, '')
    if not content.strip():
        content = 'unknown'
    return tfidf_vectorizer.transform([content]).toarray()[0]

# Gợi ý từ Ridge
def recommend(user_id=None, liked_movies=None, top_n=5):
    print(f"[RUNNING] Đang chạy recommend() cho CB_TF-IDF-Ridge với user_id={user_id}")

    if not liked_movies and user_id:
        liked_movies = get_user_history(user_id)

    matched_titles = find_matching_titles(liked_movies or [], movies['title'].tolist())
    print("[DEBUG] Matched titles:", matched_titles)

    liked_ids = movies[movies['title'].isin(matched_titles)]['movieId'].tolist()
    print("[DEBUG] liked_ids:", liked_ids)

    if not liked_ids:
        print("AAAAAAAAAAAAAAAAAAAAAAAAA")
        return ["Không tìm thấy phim phù hợp"]

    model = user_models.get(user_id)
    if not model:
        print(f"[❌] User ID {user_id} chưa được train.")
        return ["Không tìm thấy phim phù hợp"]

    watched = set(liked_ids)
    candidates = [mid for mid in movieId_to_content if mid not in watched]
    preds = {}

    printed_debug = False

    for mid in candidates:
        vec = get_movie_vector(mid).reshape(1, -1)

        if not printed_debug:
            print("[DEBUG] TF-IDF vector dim:", vec.shape[1])
            print("[DEBUG] Ridge model coef dim:", model.coef_.shape[0])
            printed_debug = True

        if vec.shape[1] != model.coef_.shape[0]:
            continue

        raw = model.predict(vec)[0]
        raw_clipped = np.clip(raw, -1, 1)
        pred_rating = ((raw_clipped + 1) / 2) * 4.5 + 0.5
        preds[mid] = pred_rating

    sorted_preds = sorted(preds.items(), key=lambda x: -x[1])[:top_n]
    recommended = [movies.loc[movies['movieId'] == mid, 'title'].values[0] for mid, _ in sorted_preds]

    print("[🎯] Gợi ý:", recommended)
    return recommended if recommended else ["Không tìm thấy phim phù hợp"]
