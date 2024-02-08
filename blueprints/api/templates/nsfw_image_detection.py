import asyncio
from PIL import Image
from transformers import pipeline
import os

classifier = pipeline("image-classification", model="Falconsai/nsfw_image_detection")

if not os.path.exists("./model"):
    classifier.save_pretrained("./model")

# Check Static Image
try:
    img = Image.open("./images/1.jpeg")
    res = classifier(img)
    print(res)
except OSError:
    print("File is not a valid image.")
