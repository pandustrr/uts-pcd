import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import base64
from io import BytesIO
import os

# === Konfigurasi Halaman ===
st.set_page_config(page_title="🧠 Restorasi Citra Digital", layout="centered")

# === Load Model ===
MODEL_PATH = os.path.join("model", "model_restorasi_citra.h5")

@st.cache_resource
def load_restore_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        st.toast("✅ Model berhasil dimuat!", icon="✅")
        return model
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        return None

model = load_restore_model()

# === Custom Tailwind Header ===
st.markdown("""
    <head>
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
""", unsafe_allow_html=True)

# === UI ===
st.markdown("""
<div class="text-center mb-6">
    <h1 class="text-2xl font-semibold text-gray-800">🧠 Restorasi Citra Digital</h1>
    <p class="text-gray-500 text-sm">Unggah gambar untuk melihat hasil restorasi citra Anda</p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 Unggah Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Tampilkan gambar asli
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<h3 class="text-center text-gray-700 mb-2">Asli</h3>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)

    with col2:
        st.markdown('<h3 class="text-center text-gray-700 mb-2">Hasil</h3>', unsafe_allow_html=True)

        if st.button("🔧 Proses Restorasi"):
            with st.spinner("⏳ Sedang memproses..."):
                try:
                    # Preprocessing
                    img = image.resize((128, 128))
                    img_array = np.array(img) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)

                    # Prediksi
                    if model:
                        restored = model.predict(img_array)[0]
                    else:
                        restored = 1 - img_array[0]  # fallback

                    # Postprocessing
                    restored = (restored * 255).astype(np.uint8)
                    restored_img = Image.fromarray(restored)

                    st.image(restored_img, use_container_width=True)

                    # Tombol download
                    buffer = BytesIO()
                    restored_img.save(buffer, format="JPEG")
                    buffer.seek(0)
                    b64 = base64.b64encode(buffer.getvalue()).decode()
                    href = f'''
                        <a href="data:file/jpg;base64,{b64}" download="hasil_restorasi.jpg"
                           class="bg-green-500 text-white px-4 py-2 rounded-lg mt-3 inline-block hover:bg-green-600 transition">
                           ⬇️ Download Hasil
                        </a>
                    '''
                    st.markdown(href, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Terjadi kesalahan: {e}")

else:
    st.markdown("""
    <div class="text-center text-gray-400 text-sm border-2 border-dashed border-gray-300 p-8 rounded-lg">
        Belum ada gambar diunggah
    </div>
    """, unsafe_allow_html=True)
