import asyncio
from PIL import Image
from transformers import pipeline
import os

classifier = pipeline("image-classification", model="Falconsai/nsfw_image_detection")


async def process_gif(gif_path):
    loop = asyncio.get_running_loop()

    with open(gif_path, "rb") as f:
        gif = await loop.run_in_executor(None, Image.open, f)

        try:
            while True:
                gif.seek(gif.tell() + 1)
                frame = gif.copy()
                result = classifier(frame)
                nsfw_found = False
                for i in result:
                    if i.get("label") == "nsfw":
                        score = i.get("score")
                        print(score)
                        if score >= 0.7:
                            nsfw_found = True
                            break
                if nsfw_found:
                    break
        except EOFError:
            pass


if not os.path.exists("./model"):
    classifier.save_pretrained("./model")

# Check Static Image
try:
    img = Image.open("./images/1.jpeg")
    res = classifier(img)
    print(res)
except OSError:
    print("File is not a valid image.")

# Check GIF
asyncio.run(process_gif("./images/gif_3.gif"))
