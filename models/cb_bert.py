import torch
import pandas as pd
import numpy as np
from utils.dataset import get_user_history
from transformers import AutoTokenizer, AutoModel

print("[DEBUG] Đang load BERT model...")
# Load model và tokenizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
bert_model = AutoModel.from_pretrained(model_name)
bert_model.eval()
bert_model.to(device)

print("[DEBUG] Đang load dữ liệu từ file pkl...")
# Load model data
model_data = torch.load('file_pkl/bert_recommender_full.pt', map_location=device)

print("[DEBUG] Các thành phần trong model:", list(model_data.keys()))

# Extract các thành phần
embedding_cache = model_data['embedding_cache']
movieId_to_content = model_data['movieId_to_content'] 
movies = model_data['movies_df']
user_train_rating = model_data['user_train_rating']

def get_bert_embedding(text):
    """Lấy embedding từ cache hoặc tính mới nếu chưa có"""
    if text in embedding_cache:
        return embedding_cache[text]
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    emb = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    embedding_cache[text] = emb
    return emb

def build_user_profile(user_movies):
    """Xây dựng profile người dùng từ danh sách phim đã xem"""
    if not user_movies:
        return None
        
    # Lấy content và embedding cho từng phim
    embeddings = []
    for movie in user_movies:
        if movie in movieId_to_content:
            content = movieId_to_content[movie]
            emb = get_bert_embedding(content)
            embeddings.append(emb)
    
    if not embeddings:
        return None
        
    # Tính trung bình các embedding
    return np.mean(embeddings, axis=0)

def compute_similarity(embedding1, embedding2):
    """Tính cosine similarity giữa 2 embedding"""
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

def recommend_topk_for_user(user_id, top_k=10):
    """
    Hàm gợi ý top-k phim cho một user sử dụng BERT embeddings
    """
    if user_id not in user_train_rating:
        print(f"❌ User {user_id} không có trong dữ liệu huấn luyện.")
        return {
            "success": False,
            "message": f"User {user_id} không có trong dữ liệu huấn luyện.",
            "recommendations": []
        }

    # Lấy danh sách phim đã xem
    train_movies = user_train_rating[user_id]
    
    # Xây dựng user profile
    user_profile = build_user_profile(train_movies)
    if user_profile is None:
        print(f"⚠️ Không thể xây dựng hồ sơ người dùng cho user {user_id}.")
        return {
            "success": False,
            "message": "Không thể xây dựng hồ sơ người dùng.",
            "recommendations": []
        }

    # Lấy danh sách phim chưa xem
    train_set = set(train_movies)
    candidate_ids = list(set(movieId_to_content.keys()) - train_set)

    # Tính similarity với tất cả phim chưa xem
    similarities = []
    for mid in candidate_ids:
        content = movieId_to_content[mid]
        movie_emb = get_bert_embedding(content)
        sim = compute_similarity(user_profile, movie_emb)
        similarities.append((mid, sim))

    # Sắp xếp và lấy top-k phim
    top_items = sorted(similarities, key=lambda x: -x[1])[:top_k]
    
    print(f"\n🎯 Top-{top_k} phim gợi ý cho user {user_id}:")
    
    recommendations = []
    for mid, sim in top_items:
        title = movies.loc[movies['movieId'] == mid, 'title'].values[0]
        print(f"{title} (similarity: {sim:.4f})")
        recommendations.append({
            "title": title,
            "similarity": float(sim),
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

    # Nếu có user_id, dùng BERT để gợi ý
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