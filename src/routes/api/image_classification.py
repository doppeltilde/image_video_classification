from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Depends
import base64
from PIL import Image
import io
from typing import List
from concurrent.futures import ThreadPoolExecutor
import filetype
from src.shared.shared import check_model, clear_cache
from src.middleware.auth.auth import get_api_key

router = APIRouter()


def classify_frame(content, i, classifier):
    img = Image.open(io.BytesIO(content))
    img.seek(i)
    result = classifier(img)
    print(f"Frame: {i} Result: {result}")
    img.close()
    return result


@router.post("/api/image-classification", dependencies=[Depends(get_api_key)])
async def image_classification(
    file: UploadFile = File(),
    model_name: str = Query(None),
):

    try:
        classifier = check_model(model_name)

        # Read the file as bytes
        contents = await file.read()
        # Check if the image is in fact an image
        if filetype.is_image(contents):
            img = Image.open(io.BytesIO(contents))
            img.verify()

            # Check if the image is a GIF and if it's animated
            if img.format.lower() == "gif":
                try:

                    results = []
                    with ThreadPoolExecutor() as executor:
                        futures = [
                            executor.submit(classify_frame, contents, i, classifier)
                            for i in range(img.n_frames)
                        ]
                        for future in futures:
                            result = future.result()
                            results.append(result)
                    return results
                except EOFError:
                    raise HTTPException(
                        status_code=400, detail="The uploaded GIF is not animated."
                    )
                finally:
                    img.close()
                    del classifier
                    clear_cache()

            # Check Static Image
            else:
                try:
                    # Validate image data
                    if not isinstance(contents, bytes):
                        raise ValueError("Invalid image data: not bytes")
                    # Encode the file to base64
                    base64Image = base64.b64encode(contents).decode("utf-8")

                    res2 = classifier(base64Image)

                    return res2
                except (ValueError, IOError) as e:
                    raise HTTPException(
                        status_code=400, detail=f"Error classifying image: {e}"
                    )
                finally:
                    img.close()
                    del classifier
                    clear_cache()
        else:
            return HTTPException(
                status_code=400, detail="The uploaded file is not a valid image."
            )

    except Exception as e:
        img.close()
        print("File is not a valid image.")
        return {"error": str(e)}

    finally:
        img.close()
        clear_cache()


@router.post("/api/multi-image-classification", dependencies=[Depends(get_api_key)])
async def multi_image_classification(
    files: List[UploadFile] = File(), model_name: str = Query(None)
):

    image_list = []

    for index, file in enumerate(files):
        try:
            classifier = check_model(model_name)

            # Read the file as bytes
            contents = await file.read()

            # Check if the image is in fact an image

            if filetype.is_image(contents):
                img = Image.open(io.BytesIO(contents))
                img.verify()

                # Check if the image is a GIF and if it's animated
                if img.format.lower() == "gif":
                    try:

                        results = []
                        with ThreadPoolExecutor() as executor:
                            futures = [
                                executor.submit(classify_frame, contents, i, classifier)
                                for i in range(img.n_frames)
                            ]
                            for future in futures:
                                result = future.result()
                                results.append(result)
                        image_list.append({index: results})
                    except EOFError:
                        raise HTTPException(
                            status_code=400, detail="The uploaded GIF is not animated."
                        )
                    finally:
                        img.close()
                        del classifier
                        clear_cache()

                # Check Static Image
                else:
                    try:
                        # Encode the file to base64
                        base64Image = base64.b64encode(contents).decode("utf-8")

                        res2 = classifier(base64Image)
                        image_list.append({index: res2})

                    except (ValueError, IOError) as e:
                        raise HTTPException(
                            status_code=400, detail=f"Error classifying image: {e}"
                        )
                    finally:
                        img.close()
                        del classifier
                        clear_cache()

            else:
                img.close()
                return HTTPException(
                    status_code=400, detail="The uploaded file is not a valid image."
                )
        except Exception as e:
            print("File is not a valid image.")
            img.close()
            return {"error": str(e)}

        finally:
            img.close()
            clear_cache()

    return image_list
