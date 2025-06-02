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
# from torch import nn
# from torch_geometric.data import Data
# from torch_geometric.nn import MessagePassing
# from torch_geometric.utils import degree

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# class LightGCNConv(MessagePassing):
#     def __init__(self):
#         super(LightGCNConv, self).__init__(aggr='add')

#     def forward(self, x, edge_index):
#         row, col = edge_index
#         deg = degree(col, x.size(0), dtype=x.dtype)
#         deg_inv_sqrt = deg.pow(-0.5)
#         deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
#         norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
#         return self.propagate(edge_index, x=x, norm=norm)

#     def message(self, x_j, norm):
#         return norm.view(-1, 1) * x_j

#     def update(self, aggr_out):
#         return aggr_out

# class LightGCN(nn.Module):
#     def __init__(self, num_users, num_items, embedding_dim=64, n_layers=3):
#         super(LightGCN, self).__init__()
#         self.num_users = num_users
#         self.num_items = num_items
#         self.embedding = nn.Embedding(num_users + num_items, embedding_dim)
#         self.convs = LightGCNConv()
#         self.n_layers = n_layers
#         nn.init.normal_(self.embedding.weight, std=0.1)

#     def forward(self, edge_index):
#         x = self.embedding.weight
#         layer_embeddings = [x]
#         for _ in range(self.n_layers):
#             x = self.convs(x, edge_index)
#             layer_embeddings.append(x)
#         embeddings = torch.stack(layer_embeddings, dim=1).mean(dim=1)
#         return embeddings[:self.num_users], embeddings[self.num_users:]

#     def predict(self, user_idx, item_idx, user_emb=None, item_emb=None):
#         if user_emb is None or item_emb is None:
#             user_emb, item_emb = self.get_embeddings()
#         user_emb = user_emb[user_idx]
#         item_emb = item_emb[item_idx - self.num_users]
#         return torch.sum(user_emb * item_emb, dim=1)

#     def get_embeddings(self):
#         with torch.no_grad():
#             return self.forward(data.edge_index)

# checkpoint = torch.load("lightgcn_full.pt", map_location=device)

# user2idx = checkpoint['user2idx']
# movie2idx = checkpoint['movie2idx']
# idx2user = checkpoint['idx2user']
# idx2movie = checkpoint['idx2movie']
# movies_df = checkpoint['movies_df']
# rating_train = checkpoint['rating_train']
# num_users = len(user2idx)
# num_movies = len(movie2idx)

# edge_user = rating_train['userId'].map(user2idx).values
# edge_movie = rating_train['movieId'].map(movie2idx).values

# edge_index = torch.tensor([
#     list(edge_user) + list(edge_movie),
#     list(edge_movie) + list(edge_user)
# ], dtype=torch.long)

# data = Data(edge_index=edge_index, num_nodes=num_users + num_movies)
# data = data.to(device)

# model = LightGCN(num_users, num_movies, embedding_dim=64, n_layers=3).to(device)
# model.load_state_dict(checkpoint['model_state_dict'])
# model.eval()

# def recommend_loaded_model(user_id, top_k=10):
#     if user_id not in user2idx:
#         print(f"❌ User {user_id} không tồn tại.")
#         return

#     u_idx = user2idx[user_id]
#     watched = set(rating_train[rating_train['userId'] == user_id]['movieId'])
#     candidates = [m for m in movie2idx if m not in watched]

#     with torch.no_grad():
#         user_emb, item_emb = model(data.edge_index)
#         user_tensor = torch.LongTensor([u_idx] * len(candidates)).to(device)
#         item_tensor = torch.LongTensor([movie2idx[m] for m in candidates]).to(device)
#         scores = model.predict(user_tensor, item_tensor, user_emb, item_emb).cpu().numpy()

#     top_scores = sorted(zip(candidates, scores), key=lambda x: -x[1])[:top_k]

#     print(f"\n🎯 Gợi ý Top-{top_k} phim cho user {user_id}:")
#     for mid, score in top_scores:
#         title = movies_df[movies_df['movieId'] == mid]['title'].values[0]
#         print(f"{title:<50} | score: {score:.4f}")
