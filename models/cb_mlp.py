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
# from torch import nn
# import numpy as np

# # ------------------- 1. Định nghĩa lại kiến trúc MLP -------------------
# class MLP(nn.Module):
#     def __init__(self, num_users, num_items, embedding_dim=64, dropout_rate=0.3):
#         super().__init__()
#         self.user_embedding = nn.Embedding(num_users, embedding_dim)
#         self.item_embedding = nn.Embedding(num_items, embedding_dim)
#         self.fc1      = nn.Linear(2 * embedding_dim, 512)
#         self.bn1      = nn.BatchNorm1d(512)
#         self.dropout1 = nn.Dropout(dropout_rate)
#         self.fc2      = nn.Linear(512, 256)
#         self.bn2      = nn.BatchNorm1d(256)
#         self.dropout2 = nn.Dropout(dropout_rate)
#         self.fc3      = nn.Linear(256, 128)
#         self.bn3      = nn.BatchNorm1d(128)
#         self.dropout3 = nn.Dropout(dropout_rate)
#         self.fc4      = nn.Linear(128, 64)
#         self.bn4      = nn.BatchNorm1d(64)
#         self.dropout4 = nn.Dropout(dropout_rate)
#         self.fc5      = nn.Linear(64, 1)
#         self.relu     = nn.ReLU()
#         self.tanh     = nn.Tanh()

#     def forward(self, user_ids, item_ids):
#         u = self.user_embedding(user_ids)
#         i = self.item_embedding(item_ids)
#         x = torch.cat([u, i], dim=1)
#         x = self.dropout1(self.relu(self.bn1(self.fc1(x))))
#         x = self.dropout2(self.relu(self.bn2(self.fc2(x))))
#         x = self.dropout3(self.relu(self.bn3(self.fc3(x))))
#         x = self.dropout4(self.relu(self.bn4(self.fc4(x))))
#         x = self.tanh(self.fc5(x))  # Output nằm trong [-1, 1]
#         return x

# # ------------------- 2. Load mô hình & metadata -------------------
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Load từ file đã lưu
# checkpoint = torch.load("mlp_full.pt", map_location=device, weights_only=False)

# user2idx = checkpoint["user2idx"]
# item2idx = checkpoint["item2idx"]
# unique_items = checkpoint["unique_items"]
# movies = checkpoint["movies_df"]
# user_train_rating = checkpoint["user_train_rating"]

# num_users = len(user2idx)
# num_items = len(item2idx)

# model = MLP(num_users, num_items).to(device)
# model.load_state_dict(checkpoint["model_state_dict"])
# model.eval()

# # ------------------- 3. Hàm gợi ý Top-K -------------------
# def recommend_topk_mlp(user_id, top_k=10):
#     if user_id not in user2idx:
#         print(f"❌ User {user_id} không tồn tại.")
#         return

#     u_idx = user2idx[user_id]
#     train_set = set(user_train_rating.get(user_id, []))
#     candidates = [m for m in unique_items if m not in train_set and m in item2idx]

#     if not candidates:
#         print(f"⚠️ User {user_id} đã xem hết các phim.")
#         return

#     user_tensor = torch.LongTensor([u_idx] * len(candidates)).to(device)
#     item_tensor = torch.LongTensor([item2idx[m] for m in candidates]).to(device)

#     with torch.no_grad():
#         scores = model(user_tensor, item_tensor).cpu().numpy().flatten()

#     top_items = sorted(zip(candidates, scores), key=lambda x: -x[1])[:top_k]

#     print(f"\n🎯 Gợi ý Top-{top_k} phim cho user {user_id}:")
#     for mid, score in top_items:
#         title = movies.loc[movies['movieId'] == mid, 'title'].values[0]
#         print(f"{title} (score: {score:.4f})")


