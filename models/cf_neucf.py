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

# import torch
# import pickle
# import pandas as pd
# import numpy as np
# from torch import nn

# # ---------------------------
# # 1. Khôi phục metadata
# # ---------------------------
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# checkpoint = torch.load('neumf_full.pt', map_location=device)

# user2idx = checkpoint['user2idx']
# movie2idx = checkpoint['movie2idx']
# idx2user = checkpoint['idx2user']
# idx2movie = checkpoint['idx2movie']
# movies_df = checkpoint['movies_df']
# rating_train = checkpoint['rating_train']

# num_users = len(user2idx)
# num_movies = len(movie2idx)

# # ---------------------------
# # 2. Khôi phục mô hình NeuMF
# # ---------------------------
# class NeuMF(nn.Module):
#     def __init__(self, num_users, num_items, embedding_dim=64):
#         super(NeuMF, self).__init__()
#         self.user_embedding_gmf = nn.Embedding(num_users, embedding_dim)
#         self.item_embedding_gmf = nn.Embedding(num_items, embedding_dim)
#         self.user_embedding_mlp = nn.Embedding(num_users, embedding_dim)
#         self.item_embedding_mlp = nn.Embedding(num_items, embedding_dim)

#         self.mlp_layers = nn.Sequential(
#             nn.Linear(2 * embedding_dim, 512),
#             nn.BatchNorm1d(512),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(512, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(256, 128),
#             nn.BatchNorm1d(128),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(128, 64),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(64, 32),
#             nn.BatchNorm1d(32),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(32, 10),
#             nn.BatchNorm1d(10),
#             nn.ReLU(),
#         )

#         self.output_layer = nn.Linear(embedding_dim + 10, 1)

#     def forward(self, user_ids, item_ids):
#         gmf_user = self.user_embedding_gmf(user_ids)
#         gmf_item = self.item_embedding_gmf(item_ids)
#         gmf_output = gmf_user * gmf_item

#         mlp_user = self.user_embedding_mlp(user_ids)
#         mlp_item = self.item_embedding_mlp(item_ids)
#         mlp_input = torch.cat([mlp_user, mlp_item], dim=-1)
#         mlp_output = self.mlp_layers(mlp_input)

#         final_input = torch.cat([gmf_output, mlp_output], dim=-1)
#         prediction = self.output_layer(final_input)
#         scaled = torch.tanh(prediction)  # giữ ở [-1, 1] để dùng trong top-K
#         return scaled.view(-1)

# # Load mô hình đã train
# model = NeuMF(num_users, num_movies).to(device)
# model.load_state_dict(torch.load('neumf_model.pt', map_location=device))
# model.eval()

# # ---------------------------
# # 3. Gợi ý Top-K phim (không cần chuyển sang [1,5])
# # ---------------------------
# def recommend_loaded_model(user_id, top_k=10):
#     if user_id not in user2idx:
#         print(f"❌ User {user_id} không tồn tại.")
#         return

#     u_idx = user2idx[user_id]
#     watched = set(rating_train[rating_train['userId'] == user_id]['movieId'])
#     candidates = [m for m in movie2idx if m not in watched]

#     user_tensor = torch.LongTensor([u_idx] * len(candidates)).to(device)
#     item_tensor = torch.LongTensor([movie2idx[m] for m in candidates]).to(device)

#     with torch.no_grad():
#         scores = model(user_tensor, item_tensor).cpu().numpy()

#     top_scores = sorted(zip(candidates, scores), key=lambda x: -x[1])[:top_k]

#     print(f"\n🎯 Gợi ý Top-{top_k} phim cho user {user_id}:")
#     for mid, score in top_scores:
#         title = movies_df[movies_df['movieId'] == mid]['title'].values[0]
#         print(f"{title:<50} | score: {score:.4f}")


