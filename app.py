from flask import Flask
from blueprints.api.nsfw_gif_detection import nsfwgif
from blueprints.api.nsfw_image_detection import nsfwimage

app = Flask(__name__)
app.register_blueprint(nsfwgif)
app.register_blueprint(nsfwimage)


@app.route("/")
def home():
    return "Flask is up and running!"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9900, debug=True)
