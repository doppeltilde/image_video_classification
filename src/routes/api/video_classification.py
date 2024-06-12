import cv2
import os
from fastapi import APIRouter, UploadFile, File, Query, Depends
from typing import List
import base64
import asyncio
from concurrent.futures import ThreadPoolExecutor
import tempfile
import filetype
from src.shared.shared import check_model, default_score
from src.middleware.auth.auth import get_api_key
import torch

router = APIRouter()

executor = ThreadPoolExecutor()


def process_video(
    classifier,
    tf,
    labels,
    score,
    fast_mode,
    skip_frames_percentage,
    return_on_first_matching_label,
):
    try:
        results = []

        # Read the video file using OpenCV
        vc = cv2.VideoCapture(tf.name)

        total_frames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))

        skip_percentage = int(total_frames * (skip_frames_percentage / 100))

        # Extract frames from the video
        while True:
            success, frame = vc.read()
            if not success:
                break
            index = int(vc.get(cv2.CAP_PROP_POS_FRAMES))

            if fast_mode and index % skip_percentage != 0:
                continue

            _, img = cv2.imencode(".jpg", frame)

            base64Image = base64.b64encode(img).decode("utf-8")
            result = classifier(base64Image)

            m = set()

            label_scores = {i["label"]: i["score"] for i in result}

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


@router.post("/api/video-classification", dependencies=[Depends(get_api_key)])
async def video_classification(
    file: UploadFile = File(),
    model_names: List[str] = Query(["Falconsai/nsfw_image_detection"], explode=True),
    return_on_first_matching_label: bool = Query(False),
    labels: List[str] = Query(["nsfw"], explode=True),
    score: float = Query(0.7),
    fast_mode: bool = Query(False),
    skip_frames_percentage: int = Query(5),
):
    try:
        totalResults = []

        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            fc = await file.read()
            tf.write(fc)

        for model_name in model_names:
            classifier = check_model(model_name)
            _score = score or default_score

            try:
                if filetype.is_video(tf.name):
                    res = await asyncio.get_event_loop().run_in_executor(
                        executor,
                        process_video,
                        classifier,
                        tf,
                        labels,
                        _score,
                        fast_mode,
                        skip_frames_percentage,
                        return_on_first_matching_label,
                    )
                    totalResults.append({model_name: res})
                else:
                    return {"error": "file is not a video"}

            except Exception as e:
                return {"error": e}

        return totalResults

    except Exception as e:
        print("File is not a valid image.")
        return {"error": str(e)}

    finally:
        tf.close()
        os.remove(tf.name)
        del classifier
        torch.cuda.empty_cache()
