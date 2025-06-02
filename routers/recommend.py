# recommend.py
from Movie_Recommendation_System_BE.models import cb_ae_mlp, cb_mlp
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from models import (
    cb_tfidf, cb_tfidf_ridge, cb_bert,
    cf_knn, cf_neucf, cf_vae, cf_lightgcn, cf_transformer
)
from utils.dataset import get_available_titles, get_movies_dataframe

router = APIRouter()

class RecommendRequest(BaseModel):
    user_id: int | None = None
    liked_movies: list[str] | None = None
    model: str | None = None  # NEW: thêm chọn model từ frontend

model_map = {
    "CB_TF-IDF": cb_tfidf,
    "CB_TF-IDF-Ridge": cb_tfidf_ridge,
    "CB_TF-IDF-MLP": cb_mlp,
    "CB_TF-IDF-AE": cb_ae_mlp,
    "CB_BERT": cb_bert,
    "CF_kNN": cf_knn,
    "CF_NeuCF": cf_neucf,
    "CF_VAE": cf_vae,
    "CF_LightGCN": cf_lightgcn,
    "CF_Transformer": cf_transformer
}

@router.post("/recommend")
def recommend_movies(req: RecommendRequest):
    if not req.model or req.model not in model_map:
        raise HTTPException(status_code=400, detail="Model không hợp lệ")

    recommender = model_map[req.model]
    print(f"[🔁] Đang gọi mô hình: {req.model}")
    return {
        "input": req.liked_movies,
        "results": recommender.recommend(user_id=req.user_id, liked_movies=req.liked_movies)
    }

@router.get("/movies")
def get_movies(limit: int = 100):
    return get_available_titles(limit)

@router.get("/movies/search")
def search_movies(query: str):
    df = get_movies_dataframe()
    results = df[df["title"].str.contains(query, case=False, na=False)]
    return results["title"].dropna().head(50).tolist()

@router.get("/movies/trending")
def trending_movies():
    df = get_movies_dataframe()
    return df["title"].dropna().sample(n=10).tolist()

@router.get("/user/history")
def user_watch_history(user_id: int):
    from utils.dataset import get_user_history
    return get_user_history(user_id)

@router.get("/user/exists")
def check_user_exists(user_id: int):
    from utils.dataset import get_ratings_dataframe
    ratings = get_ratings_dataframe()
    return {"exists": int(user_id) in ratings["userId"].unique()}
