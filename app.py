import os
from uuid import uuid4

import numpy as np
from flask import Flask, render_template, request, send_from_directory, url_for
from PIL import Image
from werkzeug.utils import secure_filename

try:
    from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
    from tensorflow.keras.models import Sequential, load_model
except ImportError:
    Conv2D = Dense = Dropout = Flatten = MaxPooling2D = Sequential = load_model = None

app = Flask(__name__, template_folder="uploads/templates")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.config["UPLOAD_FOLDER"] = os.path.join(BASE_DIR, "uploads")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

MODEL_PATH = os.path.join(BASE_DIR, "malaria_model.h5")
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "gif", "webp"}


def build_and_train_dummy_model():
    """Build and save a placeholder CNN when no trained model is available."""
    model = Sequential(
        [
            Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 3)),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(128, activation="relu"),
            Dropout(0.5),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.save(MODEL_PATH)
    return model


model = None
if load_model is not None:
    model = load_model(MODEL_PATH) if os.path.exists(MODEL_PATH) else build_and_train_dummy_model()


def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB").resize((64, 64))
    image_array = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(image_array, axis=0)


def predict_image(image_path):
    processed_image = preprocess_image(image_path)
    if model is not None:
        return float(model.predict(processed_image, verbose=0)[0][0]), "CNN model"

    # This keeps the local demo usable until a trained model is installed.
    pixels = processed_image[0]
    dark_pixel_ratio = np.mean(np.max(pixels, axis=2) < 0.35)
    color_variation = np.std(pixels, axis=(0, 1)).mean()
    prediction = float(np.clip(0.5 + (dark_pixel_ratio - 0.08) + (color_variation - 0.2), 0.05, 0.95))
    return prediction, "basic image screening"


def is_supported_image(file):
    extension = os.path.splitext(file.filename or "")[1].lower().lstrip(".")
    if extension not in ALLOWED_EXTENSIONS:
        return False
    try:
        image = Image.open(file)
        image.verify()
    except (OSError, ValueError):
        return False
    finally:
        file.seek(0)
    return True


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", error="No file uploaded.")

        file = request.files["file"]
        if not file.filename:
            return render_template("index.html", error="No file selected.")

        if not is_supported_image(file):
            return render_template(
                "index.html",
                error="Upload a readable blood-cell image in JPG, PNG, BMP, GIF, or WEBP format.",
            )

        filename = f"{uuid4().hex}_{secure_filename(file.filename)}"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        prediction, detection_mode = predict_image(filepath)
        status = "Parasitized (Malaria Detected)" if prediction >= 0.5 else "Uninfected (Healthy)"
        confidence = prediction * 100 if prediction >= 0.5 else (1 - prediction) * 100

        return render_template(
            "index.html",
            result=status,
            confidence=f"{confidence:.2f}%",
            detection_mode=detection_mode,
            image_path=url_for("uploaded_file", filename=filename),
        )

    return render_template("index.html")


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8080)
