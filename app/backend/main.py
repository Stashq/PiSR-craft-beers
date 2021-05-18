from fastapi import FastAPI, Depends, HTTPException,Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse
from DummyModel import DummyModel
import json
import pandas as pd
from fastapi import FastAPI, Header

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:5000/"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
model = DummyModel()
beer_data = pd.read_csv(f"./data/beers_metadata.csv")
ratings = pd.read_csv(f"./data/ratings.csv")
popular_beers_by_comments_count = ratings.groupby(['beer_id'],as_index=False).count().sort_values(by=["comment"]).beer_id.values
@app.get(
    "/getusers/",
    summary="Returns possible users for prototype"
)
async def get_users(request: Request):
    try:
        users =[
            {
                'name':'Ktoś1',
                'id':10
            },
            {
                'name':'Ktoś2',
                'id':11
            },
            {
                'name':'Ktoś3',
                'id':12
            },
        ]
        return json.dumps(users)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
@app.get(
    "/get_rec/",
    summary="Returns beer info for user",
)
async def get_recommendations(request: Request, user_id: int, k: int =10):
    pred = model.predict(user_id)
    recommended = pred["ForYou"]
    recommended = beer_data[beer_data.beer_id.isin(recommended)]
    best = beer_data.sort_values(by=["rating"]).head(k)
    popular = beer_data[beer_data.beer_id.isin(popular_beers_by_comments_count)].head(k)
    print(recommended)
    res = {
        'recommended':json.loads(recommended.to_json(orient="split")),
        'best': json.loads(best.to_json(orient="split")),
        'popular': json.loads(popular.to_json(orient="split")),
    }
    try:

        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

