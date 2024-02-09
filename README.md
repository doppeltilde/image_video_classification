# WIP & EXPERIMENTAL!

## Stack:
- Flask Framework
- Python
- https://huggingface.co/Falconsai/nsfw_image_detection

## Usage:
### Image Classification
Takes a base64 string.
```
curl -X POST \
     -H "Content-Type: application/json" \
     -d '{"image":""}' \
     http://localhost:9900/api/nsfw-image-detection
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