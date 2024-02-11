from fastapi import APIRouter, UploadFile, File, HTTPException
from transformers import pipeline
import os
import base64
from PIL import Image
import io
import os
from dotenv import load_dotenv

router = APIRouter()

load_dotenv()

model_name = os.getenv("MODEL_NAME", "Falconsai/nsfw_image_detection")


@router.post("/api/image-classification/")
async def nsfw_image_detection(file: UploadFile = File()):
    classifier = pipeline("image-classification", model=model_name)

    if not os.listdir(f"./models/{model_name}"):
        classifier.save_pretrained(f"./models/{model_name}")

    try:
        # Read the file as bytes
        contents = await file.read()

        # Check if the image is in fact an image
        try:
            img = Image.open(io.BytesIO(contents))
            img.verify()  # Verify that it is, in fact, an image
        except IOError:
            raise HTTPException(
                status_code=400, detail="The uploaded file is not a valid image."
            )

        # Check if the image is a GIF and if it's animated
        if img.format.lower() == "gif":
            try:
                # TODO
                print("Image is GIF")
            except EOFError:
                # If seeking to the next frame raises EOFError, it means the GIF has only one frame
                raise HTTPException(
                    status_code=400, detail="The uploaded GIF is not animated."
                )

        # Check Static Image
        else:
            # Encode the file to base64
            base64Image = base64.b64encode(contents).decode("utf-8")

            res = classifier(base64Image)
            print(res)
            return res
    except Exception as e:
        print("File is not a valid image.")
        return {"error": str(e)}
