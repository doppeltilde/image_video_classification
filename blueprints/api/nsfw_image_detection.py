from PIL import Image
from flask import Blueprint, render_template, jsonify, request
from transformers import pipeline
import os

nsfwimage = Blueprint("nsfwimage", __name__, template_folder="templates")


def add_base64_padding(base64_string):
    padding_needed = 4 - (len(base64_string) % 4)
    if padding_needed:
        base64_string += "=" * padding_needed
    return base64_string


@nsfwimage.route("/api/nsfw-image-detection/", methods=["POST"])
def app():

    classifier = pipeline(
        "image-classification", model="Falconsai/nsfw_image_detection"
    )

    if not os.path.exists("./model"):
        classifier.save_pretrained("./model")

    # Check Static Image
    try:
        image = request.get_json()["image"]
        res = classifier(image)
        print(res)
        return jsonify(res)
    except OSError:
        print("File is not a valid image.")
        return render_template("index.html")
