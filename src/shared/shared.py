from dotenv import load_dotenv
import os
from transformers import pipeline
import torch

load_dotenv()

access_token = os.getenv("ACCESS_TOKEN", None)
default_model_name = os.getenv("DEFAULT_MODEL_NAME", "Falconsai/nsfw_image_detection")
default_score = os.getenv("DEFAULT_SCORE", 0.7)
device = 0 if torch.cuda.is_available() else -1

# API KEY
api_keys_str = os.getenv("API_KEYS", "")
api_keys = api_keys_str.split(",") if api_keys_str else []
use_api_keys = os.getenv("USE_API_KEYS", "False").lower() in ["true", "1", "yes"]


def check_model(model_name):
    _model_name = model_name or default_model_name

    classifier = pipeline(
        "image-classification",
        model=_model_name,
        token=access_token,
        device=device,
    )

    classifier.save_pretrained(f"models/{_model_name}")

    return classifier
