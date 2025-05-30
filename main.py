from fastapi import FastAPI
from routers import recommend
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Cho phép gọi từ frontend React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(recommend.router)

@app.get("/")
def root():
    return {"message": "Movie recommender API is running"}
