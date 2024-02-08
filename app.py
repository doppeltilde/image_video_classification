from flask import Flask
from blueprints.api.nsfw_gif import nsfwgif

app = Flask(__name__)
app.register_blueprint(nsfwgif)


@app.route("/")
def home():
    return "Flask is up and running!"


if __name__ == "__main__":
    app.run(port=9900, debug=True)
