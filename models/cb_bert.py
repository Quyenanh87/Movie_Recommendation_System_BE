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

# import pickle
# import numpy as np
# import torch 
# from sklearn.metrics.pairwise import cosine_similarity

# # Load mô hình đã lưu
# checkpoint = torch.load("bert_recommender_full.pt")

# embedding_cache = checkpoint["embedding_cache"]
# movieId_to_content = checkpoint["movieId_to_content"]
# user_train_rating = checkpoint["user_train_rating"]
# movies = checkpoint["movies_df"]

# # Hàm lấy embedding từ cache hoặc tính lại nếu cần
# def get_bert_embedding_cached(text, tokenizer, bert_model, device):
#     if text in embedding_cache:
#         return embedding_cache[text]
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
#     with torch.no_grad():
#         outputs = bert_model(**inputs)
#     emb = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
#     embedding_cache[text] = emb
#     return emb

# # Hàm xây dựng user profile từ cache
# def build_user_profile_bert_cached(user_id, user_train_dict):
#     movie_ids = user_train_dict.get(user_id, [])
#     docs = [movieId_to_content[mid] for mid in movie_ids if mid in movieId_to_content]
#     if not docs:
#         return None
#     embeddings = [embedding_cache[doc] for doc in docs if doc in embedding_cache]
#     if not embeddings:
#         return None
#     return np.mean(embeddings, axis=0)

# # Hàm gợi ý Top-K
# def recommend_topk_bert(user_id, top_k=10):
#     if user_id not in user_train_rating:
#         print(f"❌ User {user_id} không có trong dữ liệu huấn luyện.")
#         return

#     profile = build_user_profile_bert_cached(user_id, user_train_rating)
#     if profile is None:
#         print(f"⚠️ Không thể xây dựng hồ sơ người dùng cho user {user_id}.")
#         return

#     train_set = set(user_train_rating[user_id])
#     candidate_ids = list(set(movieId_to_content.keys()) - train_set)

#     scores = []
#     for mid in candidate_ids:
#         doc = movieId_to_content[mid]
#         if doc not in embedding_cache:
#             continue
#         emb = embedding_cache[doc]
#         sim = cosine_similarity(profile.reshape(1, -1), emb.reshape(1, -1))[0][0]
#         scores.append((mid, sim))

#     top_items = sorted(scores, key=lambda x: -x[1])[:top_k]
#     print(f"\n🎯 Top-{top_k} phim gợi ý cho user {user_id}:")
#     for mid, sim in top_items:
#         title = movies[movies['movieId'] == mid]['title'].values[0]
#         print(f"{title} (similarity: {sim:.4f})")
