import cv2
import os
from fastapi import APIRouter, UploadFile, File, Query
from transformers import pipeline
from dotenv import load_dotenv
from typing import List
import base64
import asyncio
from concurrent.futures import ThreadPoolExecutor

router = APIRouter()

executor = ThreadPoolExecutor()

load_dotenv()

access_token = os.getenv("ACCESS_TOKEN", None)
default_model_name = os.getenv("DEFAULT_MODEL_NAME", "Falconsai/nsfw_image_detection")


def process_video(classifier, file, labels, score, return_on_first_matching_label):
    try:
        results = []
        tempFile = f"_temp/{file.filename}"

        with open(tempFile, "wb") as b:
            b.write(file.file.read())

        # Read the video file using OpenCV
        video_capture = cv2.VideoCapture(tempFile)

        # Extract frames from the video
        frame_count = 0
        while True:
            success, frame = video_capture.read()
            if not success:
                break

            _, img = cv2.imencode(".jpg", frame)

            base64Image = base64.b64encode(img).decode("utf-8")
            result = classifier(base64Image)

            label_scores = {i["label"]: i["score"] for i in result}

            m = set()

            for l in labels[:]:
                if l in label_scores and label_scores[l] >= score:
                    results.append(
                        {
                            "frame": frame_count,
                            "label": l,
                            "score": label_scores[l],
                        }
                    )
                    m.add(l)
                    labels.remove(l)

            if return_on_first_matching_label or set(labels) == m:
                break

            frame_count += 1

        # Release the video capture object
        video_capture.release()

        # Return the results
        return results
    except Exception as e:
        return e
    finally:
        os.remove(tempFile)


@router.post("/api/video-classification")
async def video_classification(
    file: UploadFile = File(),
    model_name: str = Query(None),
    labels: List[str] = Query(["nsfw"], explode=True),
    score: float = Query(0.7),
    return_on_first_matching_label: bool = Query(False),
):
    _model_name = model_name or default_model_name
    model_directory = f"./models/{_model_name}"

    classifier = pipeline("image-classification", model=_model_name, token=access_token)

    if not os.path.exists(model_directory):
        os.makedirs(model_directory, exist_ok=True)

    if not os.listdir(model_directory):
        classifier.save_pretrained(model_directory)

    try:
        res = await asyncio.get_event_loop().run_in_executor(
            executor,
            process_video,
            classifier,
            file,
            labels,
            score,
            return_on_first_matching_label,
        )
        print(res)
        return res

    except Exception as e:
        return {"error": e}
