from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from transformers import pipeline
import os
import base64
from PIL import Image
import io
from dotenv import load_dotenv
from typing import List
import asyncio
from concurrent.futures import ThreadPoolExecutor

router = APIRouter()

executor = ThreadPoolExecutor()

load_dotenv()

access_token = os.getenv("ACCESS_TOKEN", None)
default_model_name = os.getenv("DEFAULT_MODEL_NAME", "Falconsai/nsfw_image_detection")
default_score = os.getenv("DEFAULT_SCORE", 0.7)


def process_image(classifier, contents, labels, score, return_on_first_matching_label):
    results = []
    try:
        img = Image.open(io.BytesIO(contents))
        for frame in range(img.n_frames):
            img.seek(frame)
            result = classifier(img)
            label_scores = {i["label"]: i["score"] for i in result}

            m = set()

            for l in labels[:]:
                if l in label_scores and label_scores[l] >= score:
                    results.append(
                        {
                            "frame": frame,
                            "label": l,
                            "score": label_scores[l],
                        }
                    )
                    m.add(l)
                    labels.remove(l)

            if return_on_first_matching_label or set(labels) == m:
                break

        return results

    except Exception as e:
        print("Error processing image:", str(e))
        return results


@router.post("/api/image-query-classification")
async def image_query_classification(
    file: UploadFile = File(),
    model_name: str = Query(None),
    labels: List[str] = Query(["nsfw"], explode=True),
    score: float = Query(0.7),
    return_on_first_matching_label: bool = Query(False),
):
    _model_name = model_name or default_model_name
    _score = score or default_score
    model_directory = f"./models/{_model_name}"
    classifier = pipeline("image-classification", model=_model_name, token=access_token)

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
            img.close()
            raise HTTPException(
                status_code=400, detail="The uploaded file is not a valid image."
            )

        # Check if the image is a GIF and if it's animated
        if img.format.lower() == "gif":
            results = []

            try:
                results = await asyncio.get_event_loop().run_in_executor(
                    executor,
                    process_image,
                    classifier,
                    contents,
                    labels,
                    _score,
                    return_on_first_matching_label,
                )
                return results

            except EOFError:
                raise HTTPException(
                    status_code=400, detail="The uploaded GIF is not animated."
                )
            finally:
                img.close()

        # Check Static Image
        else:
            try:
                results = []
                # Validate image data
                if not isinstance(contents, bytes):
                    raise ValueError("Invalid image data: not bytes")
                # Encode the file to base64
                base64Image = base64.b64encode(contents).decode("utf-8")

                res = classifier(base64Image)

                label_scores = {i["label"]: i["score"] for i in res}
                for l in labels[:]:
                    if l in label_scores and label_scores[l] >= _score:
                        results.append(
                            {
                                "label": l,
                                "score": label_scores[l],
                            }
                        )

                return results
            except (ValueError, IOError) as e:
                raise HTTPException(
                    status_code=400, detail=f"Error classifying image: {e}"
                )
            finally:
                img.close()
    except Exception as e:
        img.close()
        print("File is not a valid image.")
        return {"error": str(e)}


@router.post("/api/multi-image-query-classification")
async def multi_image_query_classification(
    files: List[UploadFile] = File(),
    model_name: str = Query(None),
    labels: List[str] = Query(["nsfw"], explode=True),
    score: float = Query(0.7),
    return_on_first_matching_label: bool = Query(False),
):
    _model_name = model_name or default_model_name
    _score = score or default_score
    model_directory = f"./models/{_model_name}"
    classifier = pipeline("image-classification", model=_model_name, token=access_token)

    if not os.path.exists(model_directory):
        os.makedirs(model_directory, exist_ok=True)

    if not os.listdir(model_directory):
        classifier.save_pretrained(model_directory)

    image_list = []

    for index, file in enumerate(files):
        try:
            results = []
            labels_copy = labels.copy()

            # Read the file as bytes
            contents = await file.read()

            # Check if the image is in fact an image
            try:
                img = Image.open(io.BytesIO(contents))
                img.verify()
            except IOError:
                img.close()
                raise HTTPException(
                    status_code=400, detail="The uploaded file is not a valid image."
                )

            # Check if the image is a GIF and if it's animated
            if img.format.lower() == "gif":
                try:
                    results = await asyncio.get_event_loop().run_in_executor(
                        executor,
                        process_image,
                        classifier,
                        contents,
                        labels_copy,
                        _score,
                        return_on_first_matching_label,
                    )

                    image_list.append({index: results})

                except EOFError:
                    raise HTTPException(
                        status_code=400, detail="The uploaded GIF is not animated."
                    )
                finally:
                    img.close()

            # Check Static Image
            else:
                try:
                    results = []
                    # Validate image data
                    if not isinstance(contents, bytes):
                        raise ValueError("Invalid image data: not bytes")
                    # Encode the file to base64
                    base64Image = base64.b64encode(contents).decode("utf-8")

                    res = classifier(base64Image)

                    label_scores = {i["label"]: i["score"] for i in res}
                    for l in labels[:]:
                        if l in label_scores and label_scores[l] >= _score:
                            results.append(
                                {
                                    "label": l,
                                    "score": label_scores[l],
                                }
                            )

                    image_list.append({index: results})

                except (ValueError, IOError) as e:
                    raise HTTPException(
                        status_code=400, detail=f"Error classifying image: {e}"
                    )
                finally:
                    img.close()
        except Exception as e:
            print("File is not a valid image.")
            img.close()
            return {"error": str(e)}

    return image_list
