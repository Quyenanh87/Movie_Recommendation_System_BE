import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from utils.dataset import get_user_history

print("[DEBUG] Đang load VAE model...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model và metadata
checkpoint = torch.load('file_pkl/vae_full.pt', map_location=device, weights_only=False)
user2idx = checkpoint['user2idx']
movie2idx = checkpoint['movie2idx']
idx2movie = checkpoint['idx2movie']
movies_df = checkpoint['movies_df']
rating_train = checkpoint['rating_train']

num_users = len(user2idx)
num_movies = len(movie2idx)

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=64, dropout_rate=0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, input_dim)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

# Khởi tạo model
vae = VAE(input_dim=num_movies).to(device)
vae.load_state_dict(checkpoint['model_state_dict'])
vae.eval()

print("[DEBUG] Đã load xong VAE model")

def recommend_for_user(user_id, top_k=10):
    """Hàm gợi ý top-k phim cho một user dựa trên VAE"""
    if user_id not in user2idx:
        print(f"❌ User {user_id} không tồn tại.")
        return {
            "success": False,
            "message": f"User {user_id} không tồn tại.",
            "recommendations": []
        }

    uidx = user2idx[user_id]
    watched = set(rating_train[rating_train['userId'] == user_id]['movieId'])

    # Tạo vector input với rating gốc
    input_vector = torch.zeros(num_movies)
    ratings_user = rating_train[rating_train['userId'] == user_id]
    for row in ratings_user.itertuples():
        m_idx = movie2idx.get(row.movieId, None)
        if m_idx is not None:
            input_vector[m_idx] = row.rating

    # Điền trung bình user vào các vị trí chưa rating
    nonzero = input_vector != 0
    if nonzero.sum() > 0:
        user_mean = input_vector[nonzero].mean()
        input_vector[~nonzero] = user_mean
    else:
        input_vector[:] = 3.0  # fallback trung lập nếu user chưa đánh giá gì

    # Dự đoán
    with torch.no_grad():
        recon, _, _ = vae(input_vector.unsqueeze(0).to(device))
        recon = recon.cpu().numpy().flatten()
        recon = np.clip(recon, 1.0, 5.0)

    # Gợi ý top phim chưa xem
    candidates = [(m, recon[movie2idx[m]]) for m in movie2idx if m not in watched]
    candidates.sort(key=lambda x: -x[1])
    top = candidates[:top_k]

    print(f"\n🎯 Gợi ý Top-{top_k} phim cho user {user_id}:")
    recommendations = []
    for mid, score in top:
        title = movies_df.loc[movies_df['movieId'] == mid, 'title'].values[0]
        print(f"{title:<50} | {score:.2f}")
        recommendations.append({
            "title": title,
            "predicted_rating": float(score),
            "movieId": int(mid)
        })

    return {
        "success": True,
        "message": f"Đã tìm thấy {len(recommendations)} gợi ý phù hợp",
        "recommendations": recommendations
    }

def recommend(user_id=None, liked_movies=None, top_n=5):
    """Hàm recommend chính để tương thích với API"""
    if not liked_movies and user_id:
        liked_movies = get_user_history(user_id)
        print(f"Lịch sử phim đã xem của user {user_id}:", liked_movies)

    if not liked_movies:
        print("Không tìm thấy lịch sử xem phim")
        return ["Không tìm thấy phim phù hợp"]

    if user_id:
        print(f"\nĐang tìm gợi ý cho user {user_id}...")
        result = recommend_for_user(user_id, top_k=top_n)
        if result["success"]:
            recommendations = [rec["title"] for rec in result["recommendations"]]
            print(f"Danh sách phim gợi ý: {recommendations}")
            return recommendations
        print("Không tìm được gợi ý phù hợp")
        return ["Không tìm thấy phim phù hợp"]

    print("\nĐang xử lý danh sách phim yêu thích...")
    pool = [title for title in liked_movies if isinstance(title, str)]
    pool = list(set(pool))
    print(f"Danh sách phim hợp lệ: {pool}")

    if not pool:
        print("Không có phim hợp lệ trong danh sách")
        return ["Không tìm thấy phim phù hợp"]

    result = pool[:top_n]
    print(f"Kết quả gợi ý cuối cùng: {result}")
    return result