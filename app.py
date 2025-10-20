from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
from io import BytesIO
import base64
from tensorflow.keras.models import load_model
import os

# === Konfigurasi Aplikasi ===
app = Flask(__name__)
CORS(app)

# === Path Model ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model_restorasi_citra.h5")

# === Muat Model Restorasi ===
model = None
try:
    model = load_model(MODEL_PATH)
    print("✅ Model berhasil dimuat dari:", MODEL_PATH)
except Exception as e:
    print(f"❌ Gagal memuat model: {e}")

# === Endpoint Utama ===
@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "🧠 API Restorasi Citra Aktif!"})

# === Endpoint Proses Restorasi ===
@app.route("/restore", methods=["POST"])
def restore_image():
    """
    Menerima file gambar melalui form-data (key: 'file'),
    lalu mengembalikan hasil restorasi dalam base64.
    """
    if "file" not in request.files:
        return jsonify({"error": "Tidak ada file dikirim"}), 400

    file = request.files["file"]

    try:
        # === Baca & Preprocessing Gambar ===
        img = Image.open(file.stream).convert("RGB")
        img = img.resize((128, 128))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # === Prediksi Restorasi ===
        if model:
            restored = model.predict(img_array)[0]
        else:
            # fallback sederhana
            restored = 1 - img_array[0]

        # === Postprocessing ===
        restored = (restored * 255).astype(np.uint8)
        restored_img = Image.fromarray(restored)

        # === Konversi ke Base64 ===
        buffer = BytesIO()
        restored_img.save(buffer, format="JPEG")
        buffer.seek(0)
        encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return jsonify({
            "status": "success",
            "message": "Restorasi berhasil!",
            "restored_image": encoded_image
        })

    except Exception as e:
        print("❌ Error:", e)
        return jsonify({
            "status": "error",
            "message": f"Gagal memproses gambar: {e}"
        }), 500

# === Jalankan Server Lokal ===
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
