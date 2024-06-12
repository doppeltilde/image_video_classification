from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Depends
import base64
from PIL import Image
import io
from typing import List
import asyncio
from concurrent.futures import ThreadPoolExecutor
from src.shared.shared import check_model, default_score
from src.middleware.auth.auth import get_api_key
import torch

router = APIRouter()

executor = ThreadPoolExecutor()


def process_image(
    classifier,
    contents,
    labels,
    score,
    fast_mode,
    skip_frames_percentage,
    return_on_first_matching_label,
):
    results = []
    try:
        img = Image.open(io.BytesIO(contents))

        skip_percentage = int(img.n_frames * (skip_frames_percentage / 100))

        for frame in range(img.n_frames):
            if fast_mode and frame % skip_percentage != 0:
                continue

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

    finally:
        img.close()
        del result
        torch.cuda.empty_cache()


@router.post("/api/image-query-classification", dependencies=[Depends(get_api_key)])
async def image_query_classification(
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

        # Read the file as bytes
        contents = await file.read()

        for model_name in model_names:
            _score = score or default_score

            try:
                classifier = check_model(model_name)

                # Check if the image is in fact an image
                try:
                    img = Image.open(io.BytesIO(contents))
                    img.verify()
                except IOError:
                    img.close()
                    raise HTTPException(
                        status_code=400,
                        detail="The uploaded file is not a valid image.",
                    )

                # Check if the image is a GIF and if it's animated
                if img.format.lower() == "gif":

                    try:

                        res = await asyncio.get_event_loop().run_in_executor(
                            executor,
                            process_image,
                            classifier,
                            contents,
                            labels,
                            _score,
                            fast_mode,
                            skip_frames_percentage,
                            return_on_first_matching_label,
                        )
                        totalResults.append({model_name: res})

                    except EOFError:
                        raise HTTPException(
                            status_code=400, detail="The uploaded GIF is not animated."
                        )

                    finally:
                        img.close()
                        del classifier
                        torch.cuda.empty_cache()

                # Check Static Image
                else:
                    try:
                        results = []
                        # Validate image data
                        if not isinstance(contents, bytes):
                            raise ValueError("Invalid image data: not bytes")
                        # Encode the file to base64
                        base64Image = base64.b64encode(contents).decode("utf-8")

                        res2 = classifier(base64Image)

                        label_scores = {i["label"]: i["score"] for i in res2}
                        for l in labels[:]:
                            if l in label_scores and label_scores[l] >= _score:
                                results.append(
                                    {
                                        "label": l,
                                        "score": label_scores[l],
                                    }
                                )
                        totalResults.append({model_name: results})

                    except (ValueError, IOError) as e:
                        raise HTTPException(
                            status_code=400, detail=f"Error classifying image: {e}"
                        )

                    finally:
                        img.close()
                        del classifier
                        torch.cuda.empty_cache()

            except Exception as e:
                print("File is not a valid image.")
                return {"error": str(e)}

            finally:
                img.close()
                torch.cuda.empty_cache()

        return totalResults

    except Exception as e:
        print("File is not a valid image.")
        return {"error": str(e)}


@router.post(
    "/api/multi-image-query-classification",
    dependencies=[Depends(get_api_key)],
)
async def multi_image_query_classification(
    model_names: List[str] = Query(["Falconsai/nsfw_image_detection"], explode=True),
    files: List[UploadFile] = File(),
    return_on_first_matching_label: bool = Query(False),
    labels: List[str] = Query(["nsfw"], explode=True),
    score: float = Query(0.7),
    fast_mode: bool = Query(False),
    skip_frames_percentage: int = Query(5),
):
    _score = score or default_score

    totalResults = []

    for index, file in enumerate(files):
        # Read the file as bytes
        image_list = []

        for model_name in model_names:
            try:
                classifier = check_model(model_name)

                contents = await file.read()

                labels_copy = labels.copy()

                # Check if the image is in fact an image
                try:
                    img = Image.open(io.BytesIO(contents))
                    img.verify()
                except IOError:
                    img.close()
                    raise HTTPException(
                        status_code=400,
                        detail="The uploaded file is not a valid image.",
                    )

                # Check if the image is a GIF and if it's animated
                if img.format.lower() == "gif":
                    try:
                        res = await asyncio.get_event_loop().run_in_executor(
                            executor,
                            process_image,
                            classifier,
                            contents,
                            labels_copy,
                            _score,
                            fast_mode,
                            skip_frames_percentage,
                            return_on_first_matching_label,
                        )

                        image_list.append({model_name: res})

                    except EOFError:
                        raise HTTPException(
                            status_code=400, detail="The uploaded GIF is not animated."
                        )
                    finally:
                        img.close()
                        del classifier
                        torch.cuda.empty_cache()

                # Check Static Image
                else:
                    try:
                        results = []
                        # Validate image data
                        if not isinstance(contents, bytes):
                            raise ValueError("Invalid image data: not bytes")
                        # Encode the file to base64
                        base64Image = base64.b64encode(contents).decode("utf-8")

                        res2 = classifier(base64Image)

                        label_scores = {i["label"]: i["score"] for i in res2}
                        for l in labels[:]:
                            if l in label_scores and label_scores[l] >= _score:
                                results.append(
                                    {
                                        "label": l,
                                        "score": label_scores[l],
                                    }
                                )

                        image_list.append({model_name: results})

                    except (ValueError, IOError) as e:
                        raise HTTPException(
                            status_code=400, detail=f"Error classifying image: {e}"
                        )
                    finally:
                        img.close()
                        del classifier
                        torch.cuda.empty_cache()

            except Exception as e:
                print("File is not a valid image.")
                img.close()
                return {"error": str(e)}

            finally:
                img.close()
                torch.cuda.empty_cache()

        totalResults.append({index: image_list})

    return totalResults
