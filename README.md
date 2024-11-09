# Vision Classification for NSFW and SFW media.

## Stack:
- [FastAPI](https://fastapi.tiangolo.com)
- [Python](https://www.python.org)
- [Docker](https://docker.com)

## Installation

- For ease of use it's recommended to use the provided [docker-compose.yml](https://github.com/doppeltilde/image_video_classification/blob/main/docker-compose.yml).

### **CPU Support (AMD64 & ARM64)**
Use the `latest` tag.
```yml
services:
  image_video_classification:
    image: ghcr.io/doppeltilde/image_video_classification:latest
    ports:
      - "8000:8000"
    volumes:
      - ./models:/root/.cache/huggingface/hub:rw
    environment:
      - DEFAULT_MODEL_NAME
      - BATCH_SIZE
      - ACCESS_TOKEN
      - DEFAULT_SCORE
      - USE_API_KEYS
      - API_KEYS
    restart: unless-stopped
```

### **NVIDIA GPU Support**
> [!IMPORTANT]
> Nvidia driver >=560 needs to be installed on the host machine.

**CUDA (AMD64 & ARM64):**
```yml
services:
  image_video_classification_cuda:
    image: ghcr.io/doppeltilde/image_video_classification:latest-cuda
    ports:
      - "8000:8000"
    volumes:
      - ./models:/root/.cache/huggingface/hub:rw
    environment:
      - DEFAULT_MODEL_NAME
      - BATCH_SIZE
      - ACCESS_TOKEN
      - DEFAULT_SCORE
      - USE_API_KEYS
      - API_KEYS
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [ gpu ]
```
**OpenCL (AMD64):**
```yml
services:
  image_video_classification_opencl:
    image: ghcr.io/doppeltilde/image_video_classification:latest-opencl-nvidia
    ports:
      - "8000:8000"
    volumes:
      - ./models:/root/.cache/huggingface/hub:rw
    environment:
      - DEFAULT_MODEL_NAME
      - BATCH_SIZE
      - ACCESS_TOKEN
      - DEFAULT_SCORE
      - USE_API_KEYS
      - API_KEYS
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [ gpu ]
```

### **AMD GPU Support**
> [!WARNING]
> Unless AMD starts to officially support older Cards again (e.g. gfx8), ROCm will not be available.

**OpenCL Rusticl (AMD64):**
> [!CAUTION]
> While Rusticl is compatible with Polaris, there are significant differences in inference performance, with Polaris cards running Clover showing higher accuracy.

```yml
services:
  image_video_classification_cuda:
    image: ghcr.io/doppeltilde/image_video_classification:latest-opencl-amd-rusticl
    ports:
      - "8000:8000"
    volumes:
      - ./models:/root/.cache/huggingface/hub:rw
      - /tmp/.X11-unix:/tmp/.X11-unix
    environment:
      - DEFAULT_MODEL_NAME
      - BATCH_SIZE
      - ACCESS_TOKEN
      - DEFAULT_SCORE
      - USE_API_KEYS
      - API_KEYS
    restart: unless-stopped
    devices:
      - /dev/kfd
      - /dev/dri
    security_opt:
      - seccomp:unconfined
    group_add:
      - "video"
      - "render"
```
**OpenCL Clover (AMD64):**
```yml
services:
  image_video_classification_cuda:
    image: ghcr.io/doppeltilde/image_video_classification:latest-opencl-amd-clover
    ports:
      - "8000:8000"
    volumes:
      - ./models:/root/.cache/huggingface/hub:rw
      - /tmp/.X11-unix:/tmp/.X11-unix
    environment:
      - DEFAULT_MODEL_NAME
      - BATCH_SIZE
      - ACCESS_TOKEN
      - DEFAULT_SCORE
      - USE_API_KEYS
      - API_KEYS
    restart: unless-stopped
    devices:
      - /dev/kfd
      - /dev/dri
    security_opt:
      - seccomp:unconfined
    group_add:
      - "video"
      - "render"
```

## Environment Variables
- Create a `.env` file and set the preferred values.
```sh
DEFAULT_MODEL_NAME=Falconsai/nsfw_image_detection
BATCH_SIZE=5
DEFAULT_SCORE=0.7
ACCESS_TOKEN=

# False == Public Access
# True == Access Only with API Key
USE_API_KEYS=False

# Comma seperated api keys
API_KEYS=abc,123,xyz
```

## Tested Devices
Only tested with consumer grade hardware and only on Linux based systems.

#### CPU
- AMD FX-6300
- Intel Core i5-12400F
- AMD Ryzen 5 1600
- AMD Ryzen 5 3600
- AMD Ryzen 9 5950X

#### NVIDIA GPU (CUDA & OpenCL)
- GTX 950
- RTX 3060 Ti

#### AMD GPU (OpenCL)
- RX 580 8GB
- RX 6600 XT

#### Intel GPU (OpenCL NEO)
- None

## Models
Any model designed for image classification and compatible with huggingface transformers should work.

##### Examples
- https://huggingface.co/Falconsai/nsfw_image_detection
- https://huggingface.co/LukeJacob2023/nsfw-image-detector
- https://huggingface.co/nateraw/vit-age-classifier

## Usage

> [!NOTE]
> Please be aware that the initial classification process may require some time, as the model is being downloaded.

> [!TIP]
> Interactive API documentation can be found at: http://localhost:8000/docs

### Simple Classification

#### Image Classification
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

### Query Classification
You can utilize query parameters, if the standard classification isn't sufficient or you need a more nuanced response.

Optional parameters:
- `model_names` List[str]

- `labels` List[str]
- `score` float
- `return_on_first_matching_label` bool (default: false)

- `fast_mode` bool (default: false)
- `skip_frames_percentage` int (default: 5)

##### Single Image with Query Parameters

`POST` request to the `/api/image-query-classification` endpoint.
```sh
curl -X 'POST' \
  'http://localhost:8000/api/image-query-classification?model_names=Falconsai%2Fnsfw_image_detection' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@1.jpeg;type=image/jpeg'
```

##### Multi Image with Query Parameters
`POST` request to the `/api/multi-image-query-classification` endpoint.
```sh
curl -X 'POST' \
  'http://localhost:8000/api/multi-image-query-classification?model_names=Falconsai%2Fnsfw_image_detection' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'files=@1.png;type=image/png' \
  -F 'files=@2.jpeg;type=image/jpeg' \
  -F 'files=@3.gif;type=image/gif'
```

#### Video Classification

##### Video with Query Parameters

`POST` request to the `/api/video-classification` endpoint.

```sh
curl -X 'POST' \
  'http://localhost:8000/api/video-classification?model_names=Falconsai%2Fnsfw_image_detection' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@1.mp4;type=video/mp4'
```

> [!TIP]
> You can find code examples in the [`examples`](./examples/) folder.

> [!IMPORTANT]  
> Many thanks to Artyom Beilis and company for providing the awesome [OpenCL backend for PyTorch](https://github.com/artyom-beilis/pytorch_dlprim)!
---

_Notice:_ _This project was initally created to be used in-house, as such the
development is first and foremost aligned with the internal requirements._