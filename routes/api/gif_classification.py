# TODO
# import asyncio
# from PIL import Image
# from flask import Blueprint, render_template, jsonify, request
# from transformers import pipeline
# import os

# nsfwgif = Blueprint("nsfwgif", __name__, template_folder="templates")


# @nsfwgif.route("/api/nsfw-gif-detection/", methods=["POST"])
# def app():
#     image = request.get_json()["image"]

#     classifier = pipeline(
#         "image-classification", model="Falconsai/nsfw_image_detection"
#     )

#     if not os.path.exists("./model"):
#         classifier.save_pretrained("./model")

#     async def process_gif(img):
#         loop = asyncio.get_running_loop()

#         with open(img, "rb") as f:
#             gif = await loop.run_in_executor(None, Image.open, f)

#             try:
#                 while True:
#                     gif.seek(gif.tell() + 1)
#                     frame = gif.copy()
#                     result = classifier(frame)
#                     nsfw_found = False
#                     for i in result:
#                         if i.get("label") == "nsfw":
#                             score = i.get("score")
#                             print(score)
#                             if score >= 0.7:
#                                 nsfw_found = True
#                                 break
#                     if nsfw_found:
#                         break
#             except EOFError:
#                 pass

#     if not os.path.exists("./models"):
#         classifier.save_pretrained("./models")

#     # Check GIF
#     asyncio.run(process_gif(image))

#     return render_template("index.html")
