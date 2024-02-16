# Image & Video Classification for NSFW and SFW media.

## Stack:
- [FastAPI](https://fastapi.tiangolo.com)
- [Python](https://www.python.org)
- [Docker](https://docker.com)

## Installation
- For ease of use it's recommended to use the provided [docker-compose.yml](https://github.com/tiltedcube/image_video_classification/blob/main/docker-compose.yml).
```yml
version: '3.9'
services:
  image_video_classification:
    image: ghcr.io/tiltedcube/image_video_classification:latest
    ports:
      - "8000:8000"
    volumes:
        - models:/app/models:rw
    environment:
      - DEFAULT_MODEL_NAME
      - ACCESS_TOKEN
      - DEFAULT_SCORE
```

- Rename the `.env.example` file to `.env` and set the preferred values.

## Models
Any model designed for image classification should work.

##### Examples
- https://huggingface.co/Falconsai/nsfw_image_detection
- https://huggingface.co/LukeJacob2023/nsfw-image-detector
- https://huggingface.co/nateraw/vit-age-classifier

## Usage

### Interactive API documentation can be found at: http://localhost:8000/docs

### Image Classification
`POST` request to the `/api/image-classification` endpoint.
```sh
curl -X 'POST' \
  'http://localhost:8000/api/image-classification?model_name=Falconsai/nsfw_image_detection' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@file.jpg'
```

#### Multi Image Classification
`POST` request to the `/api/multi-image-classification` endpoint.
```sh
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

### Image Query Classification
If the standard image classification isn't sufficient or you need a more nuanced response, there are two endpoints accessible by utilizing query parameters.

Optional parameters:
- `model_name` str
- `labels` List[str]
- `score` float
- `return_on_first_matching_label` bool

#### Image with Query Parameters
`POST` request to the `/api/image-query-classification` endpoint.
```sh
curl -X 'POST' \
  'http://localhost:8000/api/image-query-classification?model_name=Falconsai%2Fnsfw_image_detection' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@1.jpeg;type=image/jpeg'
```

#### Multi Image with Query Parameters
`POST` request to the `/api/multi-image-query-classification` endpoint.
```sh
curl -X 'POST' \
  'http://localhost:8000/api/multi-image-query-classification?model_name=Falconsai%2Fnsfw_image_detection' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'files=@1.png;type=image/png' \
  -F 'files=@2.jpeg;type=image/jpeg' \
  -F 'files=@3.gif;type=image/gif'
```

Example returned json array for `LukeJacob2023/nsfw-image-detector`:
```json
[
  {
    "0": [
      {
        "label": "sexy",
        "score": 0.999847412109375
      }
    ]
  },
  {
    "1": [
      {
        "label": "porn",
        "score": 0.9998440742492676
      }
    ]
  },
  {
    "2": [
      {
        "frame": 0,
        "label": "porn",
        "score": 0.999790370464325
      },
      {
        "frame": 174,
        "label": "hentai",
        "score": 0.9852345585823059
      },
      {
        "frame": 243,
        "label": "sexy",
        "score": 0.9998371601104736
      }
    ]
  }
]
```

### Video Classification

#### Video with Query Parameters

Optional parameters:
- `model_name` str
- `labels` List[str]
- `score` float
- `return_on_first_matching_label` bool

`POST` request to the `/api/video-classification` endpoint.

```sh
curl -X 'POST' \
  'http://localhost:8000/api/video-classification?model_name=Falconsai%2Fnsfw_image_detection' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@1.mp4;type=video/mp4'
```

Example returned json array:
```json
[
  {
    "frame": 1,
    "label": "nsfw",
    "score": 0.9998898506164551
  }
]
```
---

_Notice:_ _This project was initally created to be used in-house, as such the
development is first and foremost aligned with the internal requirements._