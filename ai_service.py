"""
AI Image Detector — Microservice
=================================
سيرفيس بسيط بـ Flask بياخد صورة ويرجع لو هي حقيقية (real) أو مولّدة بالذكاء
الاصطناعي (ai)، باستخدام موديل جاهز (pretrained) من Hugging Face — من غير
أي تدريب أو تكلفة.

الموديل الافتراضي هنا: Organika/sdxl-detector
لو عايز تجرب موديل تاني بديل، غيّر قيمة MODEL_NAME لواحد من دول:
    - "Nahrawy/AIorNot"
    - "umm-maybe/AI-image-detector"

تشغيل السيرفيس:
    pip install -r requirements.txt
    python ai_service.py

هيشتغل على: http://localhost:5000
Endpoint:   POST /predict   (multipart/form-data, field name: "image")

رد الـ API:
    { "label": "real" | "ai", "confidence": 0.0-1.0, "raw": {...} }
"""

import io
import logging

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from transformers import pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "Organika/sdxl-detector"
MAX_IMAGE_SIZE_MB = 10

app = Flask(__name__)
CORS(app)  # يسمح للفرونت إنه ينادي على السيرفيس ده من دومين تاني وقت التطوير

logger.info("جاري تحميل الموديل: %s (أول مرة هياخد وقت عشان بيتنزل)...", MODEL_NAME)
classifier = pipeline("image-classification", model=MODEL_NAME)
logger.info("الموديل جاهز.")


def normalize_label(raw_label: str) -> str:
    """
    كل موديل بيرجع أسماء labels مختلفة شوية (fake/real, artificial/human,
    ai/nature...). الدالة دي بتوحّدهم لـ "ai" أو "real" عشان الفرونت
    يتعامل مع قيمة ثابتة دايمًا.
    """
    label_lower = raw_label.lower()
    ai_keywords = ["fake", "ai", "artificial", "generated", "synthetic"]
    if any(keyword in label_lower for keyword in ai_keywords):
        return "ai"
    return "real"


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": MODEL_NAME})


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "لازم ترفع صورة في field اسمه 'image'"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "الملف فاضي"}), 400

    file_bytes = file.read()
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_IMAGE_SIZE_MB:
        return jsonify({"error": f"الصورة أكبر من {MAX_IMAGE_SIZE_MB}MB"}), 400

    try:
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception:
        return jsonify({"error": "الملف ده مش صورة صحيحة"}), 400

    try:
        predictions = classifier(image)
    except Exception as exc:
        logger.exception("فشل التصنيف")
        return jsonify({"error": f"فشل تحليل الصورة: {exc}"}), 500

    # predictions شكلها: [{"label": "...", "score": 0.97}, {"label": "...", "score": 0.03}]
    top = max(predictions, key=lambda p: p["score"])
    label = normalize_label(top["label"])
    confidence = round(float(top["score"]), 4)

    return jsonify({
        "label": label,
        "confidence": confidence,
        "raw": predictions,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
