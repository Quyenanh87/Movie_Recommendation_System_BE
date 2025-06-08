import torch
import pandas as pd
import numpy as np
from torch import nn
from utils.dataset import get_user_history

print("[DEBUG] Đang load MLP model...")

# Định nghĩa kiến trúc MLP
class MLP(nn.Module):
    def __init__(self, num_users, tfidf_dim, user_embedding_dim=64, dropout_rate=0.3):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, user_embedding_dim * 4)

        self.tfidf_projection = nn.Sequential(
            nn.Linear(tfidf_dim, user_embedding_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        input_dim = user_embedding_dim * 4 * 2
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(dropout_rate)

        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.dropout4 = nn.Dropout(dropout_rate)

        self.fc5 = nn.Linear(64, 1)

    def forward(self, user_ids, tfidf_vectors):
        user_emb = self.user_embedding(user_ids)  # [B, 256]
        item_emb = self.tfidf_projection(tfidf_vectors)  # [B, 256]
        x = torch.cat([user_emb, item_emb], dim=1)
        x = self.dropout1(self.bn1(self.fc1(x)).relu())
        x = self.dropout2(self.bn2(self.fc2(x)).relu())
        x = self.dropout3(self.bn3(self.fc3(x)).relu())
        x = self.dropout4(self.bn4(self.fc4(x)).relu())
        return self.fc5(x)

# Load model và metadata
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[DEBUG] Đang load dữ liệu từ file pkl...")
checkpoint = torch.load("file_pkl/mlp_full.pt", map_location=device)

print("[DEBUG] Các thành phần trong model:", list(checkpoint.keys()))

# Extract các thành phần cần thiết
user2idx = checkpoint["user2idx"]
item2idx = checkpoint["item2idx"]
movieId_to_content_idx = checkpoint["movieId_to_content_idx"]
unique_items = checkpoint["unique_items"]
movies = checkpoint["movies_df"]
user_train_rating = checkpoint["user_train_rating"]
content_features = checkpoint["content_features"]

# Khởi tạo và load model
tfidf_dim = content_features.shape[1]
model = MLP(len(user2idx), tfidf_dim=tfidf_dim).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

def recommend_topk_for_user(user_id, top_k=10):
    """
    Hàm gợi ý top-k phim cho một user sử dụng MLP
    """
    if user_id not in user2idx:
        print(f"❌ User {user_id} không có trong dữ liệu huấn luyện.")
        return {
            "success": False,
            "message": f"User {user_id} không có trong dữ liệu huấn luyện.",
            "recommendations": []
        }

    # Lấy danh sách phim đã xem
    train_set = set(user_train_rating.get(user_id, []))
    candidates = [m for m in unique_items if m not in train_set and m in movieId_to_content_idx]
    
    if not candidates:
        print(f"⚠️ Không tìm thấy phim phù hợp cho user {user_id}.")
        return {
            "success": False,
            "message": "Không tìm thấy phim phù hợp để gợi ý.",
            "recommendations": []
        }

    # Tính điểm cho từng phim
    scores = []
    with torch.no_grad():
        u_idx = user2idx[user_id]
        user_tensor = torch.tensor([u_idx], dtype=torch.long).to(device)
        
        for mid in candidates:
            tfidf_vec = content_features[movieId_to_content_idx[mid]].unsqueeze(0).to(device)
            score = model(user_tensor, tfidf_vec).item()
            score = score * 2 + 3  # Chuyển về thang điểm [1,5] như trong Colab
            scores.append((mid, score))

    # Sắp xếp và lấy top-k phim
    top_items = sorted(scores, key=lambda x: -x[1])[:top_k]
    
    print(f"\n🎯 Top-{top_k} phim gợi ý cho user {user_id}:")
    
    recommendations = []
    for mid, score in top_items:
        title = movies.loc[movies['movieId'] == mid, 'title'].values[0]
        print(f"{title} (score: {score:.2f})")
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
    """
    Hàm recommend chính để tương thích với API
    """
    # Ưu tiên dùng lịch sử nếu không truyền liked_movies
    if not liked_movies and user_id:
        liked_movies = get_user_history(user_id)
        print("[DEBUG] Lịch sử phim đã xem:", liked_movies)

    if not liked_movies:
        print("[DEBUG] Không tìm thấy lịch sử xem phim")
        return ["Không tìm thấy phim phù hợp"]

    # Nếu có user_id, dùng MLP để gợi ý
    if user_id:
        print(f"\n[DEBUG] Đang tìm gợi ý cho user {user_id}...")
        result = recommend_topk_for_user(user_id, top_k=top_n)
        if result["success"]:
            recommendations = [rec["title"] for rec in result["recommendations"]]
            print(f"[DEBUG] Danh sách phim gợi ý: {recommendations}")
            return recommendations
        print("[DEBUG] Không tìm được gợi ý phù hợp")
        return ["Không tìm thấy phim phù hợp"]

    # Nếu không có user_id nhưng có liked_movies, trả về top_n phim từ danh sách
    print("\n[DEBUG] Đang xử lý danh sách phim yêu thích...")
    pool = [title for title in liked_movies if isinstance(title, str)]
    pool = list(set(pool))  # loại trùng
    print(f"[DEBUG] Danh sách phim hợp lệ: {pool}")
    
    if not pool:
        print("[DEBUG] Không có phim hợp lệ trong danh sách")
        return ["Không tìm thấy phim phù hợp"]
    
    result = pool[:top_n]
    print(f"[DEBUG] Kết quả gợi ý cuối cùng: {result}")
    return result