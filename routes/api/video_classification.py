import cv2
import os
from fastapi import APIRouter, UploadFile, File, Query
from transformers import pipeline
from dotenv import load_dotenv
from typing import List
import base64
import asyncio
from concurrent.futures import ThreadPoolExecutor
import tempfile
import filetype

router = APIRouter()

executor = ThreadPoolExecutor()

load_dotenv()

access_token = os.getenv("ACCESS_TOKEN", None)
default_model_name = os.getenv("DEFAULT_MODEL_NAME", "Falconsai/nsfw_image_detection")


def process_video(classifier, tf, labels, score, return_on_first_matching_label):
    try:
        results = []

        # Read the video file using OpenCV
        vc = cv2.VideoCapture(tf.name)

        # Extract frames from the video
        while True:
            success, frame = vc.read()
            if not success:
                break

            index = int(vc.get(cv2.CAP_PROP_POS_FRAMES))
            _, img = cv2.imencode(".jpg", frame)

            base64Image = base64.b64encode(img).decode("utf-8")
            result = classifier(base64Image)

            label_scores = {i["label"]: i["score"] for i in result}

            m = set()

            for l in labels[:]:
                if l in label_scores and label_scores[l] >= score:
                    results.append(
                        {
                            "frame": index,
                            "label": l,
                            "score": label_scores[l],
                        }
                    )
                    m.add(l)
                    labels.remove(l)

            if return_on_first_matching_label or set(labels) == m:
                break

        # Release the video capture object
        vc.release()

        # Return the results
        return results
    except Exception as e:
        return e
    finally:
        tf.close()
        os.remove(tf.name)


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
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            fc = file.file.read()
            tf.write(fc)

        if filetype.is_video(tf.name):
            res = await asyncio.get_event_loop().run_in_executor(
                executor,
                process_video,
                classifier,
                tf,
                labels,
                score,
                return_on_first_matching_label,
            )
            return res
        else:
            return {"error": "file is not a video"}

    except Exception as e:
        return {"error": e}
