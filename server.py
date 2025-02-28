import cv2
import numpy as np
import tensorflow as tf
import base64
import string
import json
from flask import Flask, request, jsonify
from io import BytesIO
from PIL import Image

# Danh sách ký tự trong CAPTCHA
CHARS = string.ascii_letters + string.digits
CAPTCHA_LENGTH = 5
IMG_WIDTH, IMG_HEIGHT = 128, 64

# Hàm chuyển từ nhãn sang văn bản
def label_to_text(labels):
    text = ""
    for lbl in labels:
        if lbl < 10:
            text += chr(lbl + ord('0'))
        elif lbl < 36:
            text += chr(lbl - 10 + ord('A'))
        else:
            text += chr(lbl - 36 + ord('a'))
    return text

# Load mô hình đã huấn luyện
model = tf.keras.models.load_model("captcha_model.h5")

# Hàm dự đoán CAPTCHA
def predict_captcha(image_data):
    img = Image.open(BytesIO(image_data)).convert('L')  # Chuyển về ảnh xám
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img = np.array(img) / 255.0
    img = img.reshape(1, IMG_HEIGHT, IMG_WIDTH, 1)
    
    preds = model.predict(img)
    pred_labels = [np.argmax(pred) for pred in preds]
    return label_to_text(pred_labels)

# Tạo Flask app
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        image_base64 = data.get("data")
        
        if not image_base64:
            return jsonify({"error": "No image data provided"}), 400
        
        # Giải mã base64 thành bytes
        image_data = base64.b64decode(image_base64)
        
        # Dự đoán CAPTCHA
        predicted_text = predict_captcha(image_data)
        
        return jsonify({"predicted_text": predicted_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=5000, debug=True)
