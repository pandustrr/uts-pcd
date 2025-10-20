import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import base64
from io import BytesIO
import os

# === Konfigurasi halaman ===
st.set_page_config(
    page_title="🧠 Restorasi Citra Digital",
    page_icon="🧠",
    layout="centered",
)

st.markdown("""
    <style>
    .main {
        background-color: #f9fafb;
        padding: 2rem;
        border-radius: 1rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        max-width: 800px;
        margin: auto;
    }
    h1 {
        text-align: center;
        font-weight: 600;
        color: #1f2937;
    }
    p {
        text-align: center;
        color: #6b7280;
    }
    .stButton>button {
        background-color: #3b82f6;
        color: white;
        border-radius: 0.5rem;
        padding: 0.6rem 1.2rem;
        font-weight: 500;
        border: none;
    }
    .stButton>button:hover {
        background-color: #2563eb;
    }
    </style>
""", unsafe_allow_html=True)

# === Path model ===
MODEL_PATH = os.path.join("model", "model_restorasi_citra.h5")

# === Load model dengan cache ===
@st.cache_resource
def load_restore_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

try:
    model = load_restore_model()
    st.toast("✅ Model berhasil dimuat!")
except Exception as e:
    st.error(f"❌ Gagal memuat model: {e}")
    model = None

# === UI Utama ===
with st.container():
    st.markdown('<div class="main">', unsafe_allow_html=True)

    st.title("🧠 Restorasi Citra Digital")
    st.write("Unggah gambar rusak dan lihat hasil restorasi model AI Anda!")

    uploaded_file = st.file_uploader("Pilih gambar (JPG/PNG):", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="🖼️ Gambar Asli", use_container_width=True)

        if st.button("🔧 Proses Restorasi"):
            with st.spinner("Sedang memproses..."):
                try:
                    # Preprocessing
                    img = image.resize((128, 128))
                    img_array = np.array(img) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)

                    # Prediksi
                    if model:
                        restored = model.predict(img_array)[0]
                    else:
                        restored = 1 - img_array[0]  # fallback jika model gagal dimuat

                    # Postprocessing
                    restored = (restored * 255).astype(np.uint8)
                    restored_img = Image.fromarray(restored)

                    # Tampilkan hasil
                    st.image(restored_img, caption="✨ Hasil Restorasi", use_container_width=True)

                    # Tombol download
                    buffer = BytesIO()
                    restored_img.save(buffer, format="JPEG")
                    buffer.seek(0)
                    b64 = base64.b64encode(buffer.getvalue()).decode()
                    href = f'<a href="data:file/jpg;base64,{b64}" download="restored.jpg" style="display:inline-block;margin-top:10px;background-color:#22c55e;color:white;padding:0.6rem 1rem;border-radius:0.5rem;text-decoration:none;">⬇️ Download Hasil</a>'
                    st.markdown(href, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Terjadi kesalahan: {e}")

    st.markdown('</div>', unsafe_allow_html=True)
