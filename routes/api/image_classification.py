from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from transformers import pipeline
import os
import base64
from PIL import Image
import io
from dotenv import load_dotenv
import asyncio
from typing import List

router = APIRouter()

load_dotenv()

access_token = os.getenv("ACCESS_TOKEN", None)
default_model_name = os.getenv("DEFAULT_MODEL_NAME", "Falconsai/nsfw_image_detection")


async def classify_frame(img, i, classifier):
    img.seek(i)
    frame = img.copy()
    result = classifier(frame)
    print(f"Frame: {i} Result: {result}")
    return result


@router.post("/api/image-classification")
async def nsfw_image_detection(
    file: UploadFile = File(),
    model_name: str = Query(None),
):
    model_name = model_name or default_model_name
    model_directory = f"./models/{model_name}"
    classifier = pipeline("image-classification", model=model_name, token=access_token)

    if not os.path.exists(model_directory):
        os.makedirs(model_directory, exist_ok=True)

    if not os.listdir(model_directory):
        classifier.save_pretrained(model_directory)

    try:
        # Read the file as bytes
        contents = await file.read()

        # Check if the image is in fact an image
        try:
            img = Image.open(io.BytesIO(contents))
            img.verify()
        except IOError:
            raise HTTPException(
                status_code=400, detail="The uploaded file is not a valid image."
            )

        # Check if the image is a GIF and if it's animated
        if img.format.lower() == "gif":
            try:
                loop = asyncio.get_event_loop()
                tasks = [
                    loop.create_task(classify_frame(img, i, classifier))
                    for i in range(img.n_frames)
                ]
                results = await asyncio.gather(*tasks)
                return results
            except EOFError:
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


@router.post("/api/multi-image-classification")
async def nsfw_multi_image_detection(
    files: List[UploadFile] = File(), model_name: str = Query(None)
):
    model_name = model_name or default_model_name
    model_directory = f"./models/{model_name}"
    classifier = pipeline("image-classification", model=model_name, token=access_token)

    if not os.path.exists(model_directory):
        os.makedirs(model_directory, exist_ok=True)

    if not os.listdir(model_directory):
        classifier.save_pretrained(model_directory)

    image_list = []

    for index, file in enumerate(files):
        try:
            # Read the file as bytes
            contents = await file.read()

            # Check if the image is in fact an image
            try:
                img = Image.open(io.BytesIO(contents))
                img.verify()
            except IOError:
                raise HTTPException(
                    status_code=400, detail="The uploaded file is not a valid image."
                )

            # Check if the image is a GIF and if it's animated
            if img.format.lower() == "gif":
                try:
                    loop = asyncio.get_event_loop()
                    tasks = [
                        loop.create_task(classify_frame(img, i, classifier))
                        for i in range(img.n_frames)
                    ]
                    results = await asyncio.gather(*tasks)
                    image_list.append({index: results})
                except EOFError:
                    raise HTTPException(
                        status_code=400, detail="The uploaded GIF is not animated."
                    )

            # Check Static Image
            else:
                # Encode the file to base64
                base64Image = base64.b64encode(contents).decode("utf-8")

                res = classifier(base64Image)
                image_list.append({index: res})
        except Exception as e:
            print("File is not a valid image.")
            return {"error": str(e)}

    return image_list
