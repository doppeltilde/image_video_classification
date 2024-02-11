from fastapi import APIRouter, UploadFile, File
from transformers import pipeline
import os
import base64

router = APIRouter()


@router.post("/api/image-classification/")
async def nsfw_image_detection(file: UploadFile = File()):
    classifier = pipeline(
        "image-classification", model="Falconsai/nsfw_image_detection"
    )

    if not os.listdir("./model"):
        classifier.save_pretrained("./model")

    # Check Static Image
    try:
        # Read the file as bytes
        contents = await file.read()

        # Encode the file to base64
        base64Image = base64.b64encode(contents).decode("utf-8")

        res = classifier(base64Image)
        print(res)
        return res
    except Exception as e:
        print("File is not a valid image.")
        return {"error": str(e)}
