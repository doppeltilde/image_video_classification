# NSFW & SFW Image Classification

## Stack:
- [FastAPI](https://fastapi.tiangolo.com)
- [Python](https://www.python.org)

## Installation
- For ease of use it's recommended to use the provided [docker-compose.yml](https://github.com/tiltedcube/image_classification/blob/main/docker-compose.yml).
- Rename the `.env.example` file to `.env` and set the preferred values.

## Models
Any model designed for image classification should work.

##### Examples
- https://huggingface.co/Falconsai/nsfw_image_detection
- https://huggingface.co/LukeJacob2023/nsfw-image-detector
- https://huggingface.co/nateraw/vit-age-classifier

## Usage

Interactive API documentation can be found at: http://localhost:8000/docs

#### Image Classification
`POST` request to the `/api/image-classification` endpoint, uploading an image located at file path.
```
curl -X 'POST' \
  'http://localhost:8000/api/multi-image-classification?model_name=Falconsai/nsfw_image_detection' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@file.jpg'
```

#### Multi Image Classification
`POST` request to the `/api/multi-image-classification` endpoint.
```
curl -X 'POST' \
  'http://localhost:8000/api/multi-image-classification?model_name=Falconsai/nsfw_image_detection' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'files=@0.PNG;type=image/png' \
  -F 'files=@1.JPG;type=image/jpeg' \
  -F 'files=@1.gif;type=image/gif'
```

Example returned json array for `LukeJacob2023/nsfw-image-detector`:
```json
[
      {
        "score": value,
        "label": "porn"
      },
      {
        "score": value,
        "label": "hentai"
      },
      {
        "score": value,
        "label": "sexy"
      },
      {
        "score": value,
        "label": "drawings"
      },
      {
        "score": value,
        "label": "neutral"
      }
    ]
```
---

_Notice:_ _This project was initally created to be used in-house, as such the
development is first and foremost aligned with the internal requirements._