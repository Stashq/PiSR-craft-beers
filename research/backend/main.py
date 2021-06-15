import json
from pathlib import Path
import uvicorn
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import torch
import numpy as np
from src.models import MLP
from src.util import Data
import os


MODEL_PATH = Path("../model/mlp.pt")
BEER_PATH = Path("../data/beers_metadata.csv")

assert os.path.exists(MODEL_PATH)
assert os.path.exists(BEER_PATH)

app = FastAPI()
beer_data = pd.read_csv(BEER_PATH)

origins = ["http://localhost", "http://localhost:8080", "http://localhost:5000/"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
RATINGS_PATH = Path("../data/ratings.csv")
MODEL_PATH = Path("../model/mlp.pt")

assert os.path.exists(RATINGS_PATH)
assert os.path.exists(MODEL_PATH)

data = Data(RATINGS_PATH)

device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )


user_encoder = data.user_encoder
beer_encoder = data.beer_encoder
#model = DummyModel()
model = MLP(
    user_dim=data.user_count,
    beer_dim=data.beer_count,
    n_factors=10,
    n_layers=0,
    interactions=data.train_interactions,
    user_encoder=data.user_encoder,
    beer_encoder=data.beer_encoder,
    max_rating=data.max_rating,
    learning_rate=1e-3,
    weight_decay=1e-6
)
model.load_state_dict(torch.load(MODEL_PATH))
model=model.to(device)
model.eval()


ratings = pd.read_csv(RATINGS_PATH)

popular_beer= ratings.groupby("beer_id",as_index=False).count()
beer_data= beer_data.join(popular_beer,on="beer_id",rsuffix="count_")


@app.get("/getusers/", summary="Returns possible users for prototype")
async def get_users(request: Request):
    try:
        users = [
            {"name": "Ktoś1", "id": 2201},
            {"name": "Ktoś2", "id": 1727},
            {"name": "Ktoś3", "id": 12447},
        ]
        return json.dumps(users)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

dic = {}

@app.get(
    "/get_rec/",
    summary="Returns beer info for user",
)
async def get_recommendations(request: Request, user_id: int, k: int = 10):
    print("user_id",user_id)
    print("K",k)
    beers_ids_all, rating = model.predict_ratings(user_id)

    beers_ids = beers_ids_all[:k]
    dic[str(tuple(beers_ids_all))] = 'test'
    print("test",len(dic))
    rating = rating[:k]
    print(beers_ids_all)
    print(rating)
    recommended = beer_data.set_index("beer_id").loc[beers_ids]
    recommended.rating = np.around(rating, 2)
    print(recommended)


    best = beer_data.sort_values(by=["rating"],ascending=False).head(k)


    popular = beer_data.sort_values(by=["comment"], ascending=False).head(k)

    #print(recommended)
    res = {
        "recommended": json.loads(recommended.to_json(orient="split")),
        "best": json.loads(best.to_json(orient="split")),
        "popular": json.loads(popular.to_json(orient="split")),
    }
    try:
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

