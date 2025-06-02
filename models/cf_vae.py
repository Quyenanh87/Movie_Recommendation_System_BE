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
# import pandas as pd
# import numpy as np
# import torch.nn as nn
# import torch.nn.functional as F

# # 1. Định nghĩa lại lớp VAE
# class VAE(nn.Module):
#     def __init__(self, input_dim, latent_dim=64, dropout_rate=0.3):
#         super().__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(input_dim, 1024),
#             nn.BatchNorm1d(1024),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(1024, 512),
#             nn.BatchNorm1d(512),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(512, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(256, 128),
#             nn.BatchNorm1d(128),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(128, 64),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),
#         )
#         self.fc_mu = nn.Linear(64, latent_dim)
#         self.fc_logvar = nn.Linear(64, latent_dim)
#         self.decoder = nn.Sequential(
#             nn.Linear(latent_dim, 64),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),
#             nn.Linear(64, 128),
#             nn.BatchNorm1d(128),
#             nn.ReLU(),
#             nn.Linear(128, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Linear(256, 512),
#             nn.BatchNorm1d(512),
#             nn.ReLU(),
#             nn.Linear(512, 1024),
#             nn.BatchNorm1d(1024),
#             nn.ReLU(),
#             nn.Linear(1024, input_dim)
#         )

#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return mu + eps * std

#     def forward(self, x):
#         h = self.encoder(x)
#         mu = self.fc_mu(h)
#         logvar = self.fc_logvar(h)
#         z = self.reparameterize(mu, logvar)
#         return self.decoder(z), mu, logvar

# # 2. Tải checkpoint đã lưu
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# checkpoint = torch.load('vae_full.pt', map_location=device)

# user2idx     = checkpoint['user2idx']
# movie2idx    = checkpoint['movie2idx']
# idx2movie    = checkpoint['idx2movie']
# movies_df    = checkpoint['movies_df']
# rating_train = checkpoint['rating_train']

# num_users  = len(user2idx)
# num_movies = len(movie2idx)

# # 3. Khởi tạo mô hình và load trọng số
# vae = VAE(input_dim=num_movies).to(device)
# vae.load_state_dict(checkpoint['model_state_dict'])
# vae.eval()

# def recommend_vae_loaded(user_id, top_k=10):
#     if user_id not in user2idx:
#         print(f"❌ User {user_id} không tồn tại.")
#         return

#     uidx = user2idx[user_id]

#     # Lấy danh sách phim đã xem
#     watched = set(rating_train[rating_train['userId'] == user_id]['movieId'])

#     # Chuẩn bị vector input của user
#     input_vector = torch.zeros(num_movies)
#     for row in rating_train[rating_train['userId'] == user_id].itertuples():
#         m_idx = movie2idx[row.movieId]
#         input_vector[m_idx] = (row.rating - 3) / 2  # normalize về [-1,1]

#     with torch.no_grad():
#         recon, _, _ = vae(input_vector.unsqueeze(0).to(device))
#         recon = recon.cpu().numpy().flatten()
#         recon = (recon + 1) * 2 + 1  # đưa về [1,5]
#         recon = np.clip(recon, 1.0, 5.0)

#     # Gợi ý top phim chưa xem
#     candidates = [(m, recon[movie2idx[m]]) for m in movie2idx if m not in watched]
#     candidates.sort(key=lambda x: -x[1])
#     top = candidates[:top_k]

#     print(f"\n🎯 Gợi ý Top-{top_k} phim cho user {user_id}:")
#     for mid, score in top:
#         title = movies_df.loc[movies_df['movieId'] == mid, 'title'].values[0]
#         print(f"{title:<50} | {score:.2f}")
