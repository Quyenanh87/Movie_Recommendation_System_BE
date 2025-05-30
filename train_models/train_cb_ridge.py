import sys
import os
import pickle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import pandas as pd
import math
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from utils.dataset import get_movies_dataframe, get_ratings_dataframe

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Đường dẫn lưu mô hình
model_path = "file_pkl/user_models_cb_tfidf_ridge.pkl"
vectorizer_path = "file_pkl/tfidf_vectorizer_cb_tfidf_ridge.pkl"

# Nếu tồn tại file cũ thì xóa
if os.path.exists(model_path):
    os.remove(model_path)
if os.path.exists(vectorizer_path):
    os.remove(vectorizer_path)

print("[🔁] Loading data...")
movies = get_movies_dataframe()
ratings = get_ratings_dataframe()

# Tiền xử lý content
movies['content'] = (movies['title'].fillna('') + ' ' + movies['genres'].fillna('')).str.replace('|', ' ').str.lower()
movieId_to_content = dict(zip(movies['movieId'], movies['content']))

# Normalize rating về [-1, 1]
ratings['rating_norm'] = ((ratings['rating'] - 0.5) / 4.5) * 2 - 1

print("[🔤] Fitting TF-IDF vectorizer...")
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
tfidf_vectorizer.fit(movies['content'].values)

# Hàm lấy vector của phim
def get_movie_vector(mid):
    content = movieId_to_content.get(mid, '')
    return tfidf_vectorizer.transform([content]).toarray()[0]

print("[🏗️] Training Ridge models for users...")
user_models = {}
user_groups = ratings.groupby('userId')

for user_id, group in user_groups:
    if len(group) < 5:
        continue
    X, y = [], []
    for _, row in group.iterrows():
        mid = row['movieId']
        vec = get_movie_vector(mid)
        X.append(vec)
        y.append(row['rating_norm'])
    model = Ridge(alpha=1.0)
    model.fit(X, y)
    user_models[user_id] = model

print(f"[✅] Trained Ridge models for {len(user_models)} users")

# ==== ĐÁNH GIÁ MÔ HÌNH ====
print("[📊] Đang đánh giá mô hình...")

# Tách tập test: lấy 1 tương tác cuối mỗi user làm test
valid_users = ratings['userId'].value_counts()
valid_users = valid_users[valid_users > 5].index
filtered = ratings[ratings['userId'].isin(valid_users)]
test_set = filtered.groupby('userId').tail(1)
train_set = ratings.drop(test_set.index)

user_test = test_set.groupby('userId')['movieId'].apply(list).to_dict()
test_ratings = test_set.groupby('userId')['rating'].apply(list).to_dict()
user_train = train_set.groupby('userId')['movieId'].apply(list).to_dict()

# Evaluate rating
y_true_all, y_pred_all = [], []
for user in user_test.keys():
    if user not in user_models:
        continue
    model = user_models[user]
    test_ids = user_test[user]
    for idx, mid in enumerate(test_ids):
        if mid not in movieId_to_content:
            continue
        vec = get_movie_vector(mid).reshape(1, -1)
        raw = model.predict(vec)[0]
        raw = np.clip(raw, -1, 1)
        pred = ((raw + 1) / 2) * 4.5 + 0.5
        y_pred_all.append(pred)
        y_true_all.append(test_ratings.get(user, [])[idx])

mae = mean_absolute_error(y_true_all, y_pred_all)
mse = mean_squared_error(y_true_all, y_pred_all)
rmse = np.sqrt(mse)

print(f"[🎯] Rating Evaluation: MAE={mae:.4f}, MSE={mse:.4f}, RMSE={rmse:.4f}")

# Evaluate top-K
K = 10
precision_list, hr_list, ndcg_list = [], [], []
all_movie_ids = set(movieId_to_content.keys())

for user in user_test.keys():
    if user not in user_models:
        continue
    model = user_models[user]
    pos_items = user_test.get(user, [])
    if not pos_items:
        continue
    pos_mid = pos_items[0]

    train_items = set(user_train.get(user, []))
    candidates = list(all_movie_ids - train_items)
    if pos_mid not in candidates:
        continue

    negatives = list(set(candidates) - {pos_mid})
    if len(negatives) >= 49:
        negatives = random.sample(negatives, 49)
    items = negatives + [pos_mid]

    preds = {}
    for mid in items:
        vec = get_movie_vector(mid).reshape(1, -1)
        raw = model.predict(vec)[0]
        raw = np.clip(raw, -1, 1)
        pred = ((raw + 1) / 2) * 4.5 + 0.5
        preds[mid] = pred

    ranked = sorted(preds.items(), key=lambda x: -x[1])
    ranked_ids = [mid for mid, _ in ranked]
    rank = ranked_ids.index(pos_mid) + 1 if pos_mid in ranked_ids else np.inf

    precision = 1.0 / K if rank <= K else 0.0
    hr = 1.0 if rank <= K else 0.0
    ndcg = 1.0 / math.log2(rank + 1) if rank <= K else 0.0

    precision_list.append(precision)
    hr_list.append(hr)
    ndcg_list.append(ndcg)

avg_precision = np.mean(precision_list)
avg_hr = np.mean(hr_list)
avg_ndcg = np.mean(ndcg_list)

print(f"[📌] Top-{K} Evaluation: Precision@{K}={avg_precision:.4f}, HR@{K}={avg_hr:.4f}, NDCG@{K}={avg_ndcg:.4f}")

# ==== LƯU FILE ====
output_dir = "file_pkl"
os.makedirs(output_dir, exist_ok=True)

print("[💾] Saving user models to file_pkl/user_models_cb_tfidf_ridge.pkl")
with open(model_path, "wb") as f:
    pickle.dump(user_models, f)

print("[💾] Saving TF-IDF vectorizer to file_pkl/tfidf_vectorizer_cb_tfidf_ridge.pkl")
with open(vectorizer_path, "wb") as f:
    pickle.dump(tfidf_vectorizer, f)

print("[🎉] Training complete. Models saved!")
