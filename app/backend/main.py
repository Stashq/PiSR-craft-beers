from fastapi import FastAPI, Depends, HTTPException,Request
from fastapi.middleware.cors import CORSMiddleware
import json

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

@app.get(
    "/getusers/",
    summary="Returns possible users for prototype"
)
async def get_users(request: Request):
    try:
        users =[
            {
                'name':'1',
                'id':10
            },
            {
                'name':'1',
                'id':10
            },
            {
                'name':'1',
                'id':10
            },
        ]
        return json.dumps(users)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))