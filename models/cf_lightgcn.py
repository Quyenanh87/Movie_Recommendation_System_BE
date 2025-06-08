import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
from torch_geometric.data import Data
from utils.dataset import get_user_history

print("[DEBUG] Đang load LightGCN model...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model và metadata
checkpoint = torch.load('file_pkl/lightgcn_full.pt', map_location=device, weights_only=False)
user2idx = checkpoint['user2idx']
movie2idx = checkpoint['movie2idx']
idx2user = checkpoint['idx2user']
idx2movie = checkpoint['idx2movie']
movies_df = checkpoint['movies_df']
rating_train = checkpoint['rating_train']

num_users = len(user2idx)
num_movies = len(movie2idx)

# Chuẩn bị dữ liệu đồ thị
edge_user = rating_train['userId'].map(user2idx).values
edge_movie = rating_train['movieId'].map(movie2idx).values

edge_index = torch.tensor([
    list(edge_user) + list(edge_movie),
    list(edge_movie) + list(edge_user)
], dtype=torch.long)

data = Data(edge_index=edge_index, num_nodes=num_users + num_movies)
data = data.to(str(device))  # Convert device to string

class LightGCNConv(MessagePassing):
    def __init__(self):
        super(LightGCNConv, self).__init__(aggr='add')

    def forward(self, x, edge_index):
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm, index):  # Add index parameter
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out, inputs=None):  # Add inputs parameter
        return aggr_out

class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64, n_layers=3):
        super(LightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding = nn.Embedding(num_users + num_items, embedding_dim)
        self.convs = LightGCNConv()
        self.n_layers = n_layers
        nn.init.normal_(self.embedding.weight, std=0.1)

    def forward(self, edge_index):
        x = self.embedding.weight
        layer_embeddings = [x]
        for _ in range(self.n_layers):
            x = self.convs(x, edge_index)
            layer_embeddings.append(x)
        embeddings = torch.stack(layer_embeddings, dim=1).mean(dim=1)
        return embeddings[:self.num_users], embeddings[self.num_users:]

    def predict(self, user_idx, item_idx, user_emb=None, item_emb=None):
        if user_emb is None or item_emb is None:
            user_emb, item_emb = self.get_embeddings()
        user_emb = user_emb[user_idx]
        item_emb = item_emb[item_idx - self.num_users]
        return torch.sum(user_emb * item_emb, dim=1)

    def get_embeddings(self):
        with torch.no_grad():
            return self.forward(data.edge_index)

# Khởi tạo model và tính sẵn embeddings
model = LightGCN(num_users, num_movies, embedding_dim=64, n_layers=3).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Cache embeddings để tái sử dụng
with torch.no_grad():
    cached_user_emb, cached_item_emb = model(data.edge_index)

print("[DEBUG] Đã load xong LightGCN model")

def recommend_for_user(user_id, top_k=10):
    """Hàm gợi ý top-k phim cho một user dựa trên LightGCN"""
    if user_id not in user2idx:
        print(f"❌ User {user_id} không tồn tại.")
        return {
            "success": False,
            "message": f"User {user_id} không tồn tại.",
            "recommendations": []
        }

    u_idx = user2idx[user_id]
    watched = set(rating_train[rating_train['userId'] == user_id]['movieId'])
    
    # Lấy ngẫu nhiên một tập con các phim chưa xem để tăng tốc độ
    candidates = list(set(movie2idx.keys()) - watched)
    if len(candidates) > 1000:  # Chỉ lấy 1000 phim ngẫu nhiên nếu có quá nhiều
        candidates = np.random.choice(candidates, 1000, replace=False)

    # Tính toán scores theo batch
    batch_size = 256
    all_scores = []
    
    with torch.no_grad():
        user_emb = cached_user_emb[u_idx].unsqueeze(0)  # [1, dim]
        
        for i in range(0, len(candidates), batch_size):
            batch_items = candidates[i:i + batch_size]
            item_indices = torch.tensor([movie2idx[m] for m in batch_items], device=device)
            batch_item_emb = cached_item_emb[item_indices - num_users]  # [batch, dim]
            
            # Tính similarity cho cả batch
            batch_scores = torch.mm(user_emb, batch_item_emb.t()).squeeze()  # [batch]
            all_scores.append(batch_scores)
    
    # Ghép các scores lại
    scores = torch.cat(all_scores).cpu().numpy()
    
    # Lấy top-k
    top_indices = np.argsort(-scores)[:top_k]
    top_items = [candidates[i] for i in top_indices]
    top_scores = scores[top_indices]

    print(f"\n🎯 Gợi ý Top-{top_k} phim cho user {user_id}:")
    recommendations = []
    for mid, score in zip(top_items, top_scores):
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