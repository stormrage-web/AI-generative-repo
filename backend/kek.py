import json

from fastapi import FastAPI, File
import uvicorn

from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Annotated
from fastapi.middleware.cors import CORSMiddleware

import cv2
import numpy as np
import requests
from urllib.parse import unquote

from PIL import Image
from io import BytesIO

import uuid

from ml import REMOVE_INFOGRAPHICS, REMOVE_BACKGR


class PromptRequest(BaseModel):
    url: str
    prompt: str

class ActionRequest(BaseModel):
    url: str
    type: str


app = FastAPI()

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/source")
async def source(files: str):
    print(json.loads(unquote(files)))
    return JSONResponse([
        {
            "id": 8800553535,
            "url": "https://pofoto.club/uploads/posts/2022-08/1660343110_37-pofoto-club-p-zimorodok-foto-zimoi-44.jpg",
            "generated": [
                {
                    "id": 101,
                    "url": "https://pofoto.club/uploads/posts/2022-08/1660343110_37-pofoto-club-p-zimorodok-foto-zimoi-44.jpg"
                },
            ]
        },
    ])

@app.post("/prompt")
async def prompt(prompt_images: PromptRequest):
    return [
        {
            "id": 8800553535,
            "url": "https://pofoto.club/uploads/posts/2022-08/1660343110_37-pofoto-club-p-zimorodok-foto-zimoi-44.jpg"
        },
    ]


@app.post("/action")
async def action(action_images: ActionRequest):
    saved_id = str(uuid.uuid4())
    if action_images.type == "info":
        resp = requests.get(action_images.url, stream=True).raw
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        no_info = REMOVE_INFOGRAPHICS(image)
        cv2.imwrite(f"images/{saved_id}.png", no_info)
    if action_images.type == "white":
        response = requests.get(action_images.url)
        img = Image.open(BytesIO(response.content)).convert('RGB')
        no_info = REMOVE_BACKGR(img)
        cv2.imwrite(f"images/{saved_id}.png", no_info)
    return [
        {
            "id": 8800553535,
            "url": f"http://51.250.91.130:8000/{saved_id}.png"
        },
    ]

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000, log_level="info")
