from fastapi import FastAPI
from routes.api import image_classification
from routes.api import image_query_classification
from routes.api import video_classification

app = FastAPI()
app.include_router(image_classification.router)
app.include_router(image_query_classification.router)
app.include_router(video_classification.router)


@app.get("/")
def root():
    return {"res": "FastAPI is up and running!"}
