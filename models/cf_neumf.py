import torch
import torch.nn as nn
import numpy as np
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model và metadata
checkpoint = torch.load('file_pkl/neumf.pt', map_location=device, weights_only=False)
user2idx = checkpoint['user2idx']
movie2idx = checkpoint['movie2idx']
idx2user = checkpoint['idx2user']
idx2movie = checkpoint['idx2movie']
movies_df = checkpoint['movies_df']
rating_train = checkpoint['rating_train']

num_users = len(user2idx)
num_movies = len(movie2idx)

class NeuMF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64):
        super(NeuMF, self).__init__()
        self.user_embedding_gmf = nn.Embedding(num_users, embedding_dim)
        self.item_embedding_gmf = nn.Embedding(num_items, embedding_dim)
        self.user_embedding_mlp = nn.Embedding(num_users, embedding_dim)
        self.item_embedding_mlp = nn.Embedding(num_items, embedding_dim)

        self.mlp_layers = nn.Sequential(
            nn.Linear(2 * embedding_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 10),
            nn.BatchNorm1d(10),
            nn.ReLU(),
        )

        self.output_layer = nn.Linear(embedding_dim + 10, 1)

    def forward(self, user_ids, item_ids):
        gmf_user = self.user_embedding_gmf(user_ids)
        gmf_item = self.item_embedding_gmf(item_ids)
        gmf_output = gmf_user * gmf_item

        mlp_user = self.user_embedding_mlp(user_ids)
        mlp_item = self.item_embedding_mlp(item_ids)
        mlp_input = torch.cat([mlp_user, mlp_item], dim=-1)
        mlp_output = self.mlp_layers(mlp_input)

        final_input = torch.cat([gmf_output, mlp_output], dim=-1)
        prediction = self.output_layer(final_input)
        return prediction.view(-1)

# Khởi tạo model
model = NeuMF(num_users, num_movies).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

def recommend_for_user(user_id, top_k=10):
    """Hàm gợi ý top-k phim cho một user dựa trên NeuMF"""
    if user_id not in user2idx:
        print(f"❌ User {user_id} không tồn tại.")
        return {
            "success": False,
            "message": f"User {user_id} không tồn tại.",
            "recommendations": []
        }

    u_idx = user2idx[user_id]
    watched = set(rating_train[rating_train['userId'] == user_id]['movieId'])
    candidates = [m for m in movie2idx.keys() if m not in watched]

    if not candidates:
        return {
            "success": False,
            "message": "Không tìm thấy phim phù hợp để gợi ý.",
            "recommendations": []
        }

    with torch.no_grad():
        user_tensor = torch.LongTensor([u_idx] * len(candidates)).to(device)
        item_tensor = torch.LongTensor([movie2idx[m] for m in candidates]).to(device)
        scores = model(user_tensor, item_tensor).cpu().numpy()

    movie_scores = list(zip(candidates, scores))
    top_scores = sorted(movie_scores, key=lambda x: -x[1])[:top_k]

    print(f"\n🎯 Gợi ý Top-{top_k} phim cho user {user_id}:")
    recommendations = []
    for mid, score in top_scores:
        title = movies_df[movies_df['movieId'] == mid]['title'].values[0]
        print(f"{title:<50} | score: {score:.4f}")
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
    from utils.dataset import get_user_history

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