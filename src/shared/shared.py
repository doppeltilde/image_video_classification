from dotenv import load_dotenv
import os

# from transformers import pipeline
# from transformers import AutoFeatureExtractor
# from optimum.pipelines import pipeline
# from optimum.onnxruntime import ORTModelForImageClassification
import torch
import requests
from PIL import Image
from transformers import AutoFeatureExtractor, pipeline
from optimum.onnxruntime import ORTModelForImageClassification

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
    _model_name = "AdamCodd/vit-base-nsfw-detector"  # model_name or default_model_name

    preprocessor = AutoFeatureExtractor.from_pretrained(_model_name)
    model = ORTModelForImageClassification.from_pretrained(
        _model_name,
        file_name="onnx/model_quantized.onnx",
    )
    onnx_image_classifier = pipeline(
        "image-classification", model=model, feature_extractor=preprocessor
    )

    return onnx_image_classifier


# preprocessor = AutoFeatureExtractor.from_pretrained(_model_name)
# model = ORTModelForImageClassification.from_pretrained(_model_name, export=True)

# classifier = pipeline(
#     "image-classification",
#     model=model,
#     token=access_token,
#     device=device,
#     feature_extractor=preprocessor,
#     accelerator="ort",
# )

# return classifier
