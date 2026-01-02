from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
import os

# ==============================
# CONFIG
# ==============================
IMAGE_SIZE = 224
UPLOAD_FOLDER = "static"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ==============================
# LOAD MODEL + CLASS NAMES
# ==============================
model = tf.keras.models.load_model("my_model.keras")
print("FLASK MODEL INPUT SHAPE:", model.input_shape)

with open("class_names.txt") as f:
    class_names = [line.strip() for line in f]

print("FLASK CLASS ORDER:", class_names)

# ==============================
# UTILS
# ==============================
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def predict(img):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = tf.expand_dims(img_array, 0)

    preds = model.predict(img_array)
    idx = int(np.argmax(preds[0]))

    return class_names[idx], round(float(np.max(preds[0]) * 100), 2)

# ==============================
# ROUTES
# ==============================
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":

        if "file" not in request.files:
            return render_template("index.html", message="No file uploaded")

        file = request.files["file"]

        if file.filename == "":
            return render_template("index.html", message="No file selected")

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(path)

            img = tf.keras.preprocessing.image.load_img(
                path, target_size=(224, 224)
            )

            label, confidence = predict(img)

            return render_template(
                "index.html",
                image_path=path,
                predicted_label=label,
                confidence=confidence
            )

    return render_template("index.html")

# ==============================
# RUN
# ==============================
if __name__ == "__main__":
    app.run(debug=True)
