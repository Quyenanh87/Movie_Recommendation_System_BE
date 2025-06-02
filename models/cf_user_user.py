# import pickle
# import numpy as np
# import pandas as pd

# # Load mô hình đã lưu
# with open("usercf_model.pkl", "rb") as f:
#     model = pickle.load(f)

# sim_df = model["sim_df"]
# train_rating_dict = model["train_rating_dict"]
# user_avg_rating = model["user_avg_rating"]
# user_train_rating = model["user_train_rating"]
# movies_df = model["movies_df"]

# # Hàm dự đoán
# def predict_rating_user_user(user, movie, user_train_dict, sim_df, train_rating_dict, user_avg_rating, neighbor=10):
#     users_who_rated = [v for (v, m) in train_rating_dict.keys() if m == movie and v != user]
#     if len(users_who_rated) == 0 or user not in sim_df.index:
#         return user_avg_rating.get(user, np.nan)
#     sim_list = []
#     for v in users_who_rated:
#         if v in sim_df.columns:
#             sim = sim_df.at[user, v]
#             r_val = train_rating_dict.get((v, movie))
#             if r_val is not None and not np.isnan(sim):
#                 diff = r_val - user_avg_rating.get(v, 0)
#                 sim_list.append((sim, diff))
#     if len(sim_list) == 0:
#         return user_avg_rating.get(user, np.nan)
#     sim_list = sorted(sim_list, key=lambda x: abs(x[0]), reverse=True)[:neighbor]
#     similarities = np.array([s for s, _ in sim_list])
#     rating_diffs = np.array([d for _, d in sim_list])
#     weighted_sum = np.dot(similarities, rating_diffs)
#     sum_sim = np.sum(np.abs(similarities))
#     pred_diff = weighted_sum / sum_sim if sum_sim != 0 else 0
#     pred = user_avg_rating.get(user, 0) + pred_diff
#     return np.clip(pred, 1, 5)

# # Hàm gợi ý Top-K cho user
# def recommend_topk_usercf(user_id, top_k=10, neighbor=10):
#     if user_id not in user_train_rating:
#         print(f"❌ User {user_id} không có dữ liệu train.")
#         return

#     watched = set(user_train_rating[user_id])
#     all_movies = set([m for (_, m) in train_rating_dict.keys()])
#     candidates = list(all_movies - watched)

#     preds = []
#     for m in candidates:
#         pred = predict_rating_user_user(user_id, m, user_train_rating, sim_df, train_rating_dict, user_avg_rating, neighbor)
#         if not np.isnan(pred):
#             preds.append((m, pred))

#     top_items = sorted(preds, key=lambda x: -x[1])[:top_k]
#     print(f"\n🎯 Top {top_k} phim gợi ý cho user {user_id} (User-User CF):")
#     for movie_id, score in top_items:
#         title = movies_df[movies_df['movieId'] == movie_id]['title'].values[0]
#         print(f"{title} — predicted rating: {score:.2f}")
