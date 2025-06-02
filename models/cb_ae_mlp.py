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

# # Định nghĩa lại lớp AutoencoderMLP
# class AutoencoderMLP(nn.Module):
#     def __init__(self, num_users, num_items, embedding_dim=64, dropout_rate=0.3):
#         super().__init__()
#         self.user_embedding = nn.Embedding(num_users, embedding_dim * 4)
#         self.item_embedding = nn.Embedding(num_items, embedding_dim * 4)
#         self.encoder = nn.Sequential(
#             nn.Linear(embedding_dim * 8, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate),
#             nn.Linear(256, 64),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),
#         )
#         self.decoder = nn.Sequential(
#             nn.Linear(64, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate),
#             nn.Linear(256, 512),
#             nn.BatchNorm1d(512),
#             nn.ReLU(),
#         )
#         self.predictor = nn.Sequential(
#             nn.Linear(64, 32),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate),
#             nn.Linear(32, 1),
#             nn.Tanh()
#         )

#     def forward(self, user_ids, item_ids):
#         user_emb = self.user_embedding(user_ids)
#         item_emb = self.item_embedding(item_ids)
#         x = torch.cat([user_emb, item_emb], dim=1)
#         latent = self.encoder(x)
#         rating_pred = self.predictor(latent)
#         recon = self.decoder(latent)
#         target = x.detach()
#         return rating_pred, recon, target

# # Load lại mô hình
# # Tải lại toàn bộ
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# checkpoint = torch.load("autoencoder_full.pt", map_location=device)

# # Khôi phục metadata
# user2idx = checkpoint["user2idx"]
# item2idx = checkpoint["item2idx"]
# unique_items = checkpoint["unique_items"]
# movies = checkpoint["movies_df"]
# user_train_rating = checkpoint["user_train_rating"]

# # Khôi phục model
# num_users = len(user2idx)
# num_items = len(item2idx)
# model = AutoencoderMLP(num_users, num_items).to(device)
# model.load_state_dict(checkpoint["model_state_dict"])
# model.eval()

# # Gợi ý phim cho user_id
# def recommend_topk_autoencoder(user_id, top_k=10):
#     if user_id not in user2idx:
#         print(f"❌ User {user_id} không có trong dữ liệu.")
#         return

#     ui = torch.tensor(user2idx[user_id], dtype=torch.long).unsqueeze(0).to(device)
#     train_set = set(user_train_rating.get(user_id, []))
#     candidates = [m for m in unique_items if m not in train_set]

#     scores = []
#     with torch.no_grad():
#         for m in candidates:
#             if m not in item2idx:
#                 continue
#             mi = torch.tensor(item2idx[m], dtype=torch.long).unsqueeze(0).to(device)
#             score = model(ui, mi)[0].item()  # đầu ra [-1,1]
#             scores.append((m, score))

#     ranked = sorted(scores, key=lambda x: -x[1])[:top_k]
#     print(f"\n🎯 Gợi ý Top-{top_k} phim cho user {user_id}:")
#     for mid, score in ranked:
#         title = movies.loc[movies['movieId'] == mid, 'title'].values[0]
#         print(f"{title} (score: {score:.4f})")

