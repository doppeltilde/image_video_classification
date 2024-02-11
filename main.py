from fastapi import FastAPI
from routes.api import image_classification

app = FastAPI()
app.include_router(image_classification.router)


@app.get("/")
def root():
    return {"res": "FastAPI is up and running!"}
