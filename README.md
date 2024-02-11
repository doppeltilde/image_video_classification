# WIP & EXPERIMENTAL!

## Stack:
- [FastAPI](https://fastapi.tiangolo.com)
- [Python](https://www.python.org)
- https://huggingface.co/Falconsai/nsfw_image_detection

## Usage:

Interactive API documentation can be found at: http://localhost:8000/docs

### Image Classification
`POST` request to the `/api/image-classification/` endpoint, uploading an image located at file path.
```
curl -X POST \
    -H "Content-Type: multipart/form-data" \
    -F "file=@file.jpg" \
    http://localhost:8000/api/image-classification/

```
Returns a json array.
```json
[
	{
		"label": "nsfw",
		"score": value
	},
	{
		"label": "normal",
		"score": value
	}
]
```
---

_Notice:_ _This project was initally created to be used in-house, as such the
development is first and foremost aligned with the internal requirements._