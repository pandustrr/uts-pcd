import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import base64
from io import BytesIO
import os

# === Load Model ===
MODEL_PATH = os.path.join("model", "model_restorasi_citra.h5")

@st.cache_resource
def load_restore_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        st.success("✅ Model berhasil dimuat!")
        return model
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        return None

model = load_restore_model()

# === UI ===
st.title("🧠 Aplikasi Restorasi Citra")
st.write("Unggah gambar rusak, dan model akan mencoba merestorasinya.")

uploaded_file = st.file_uploader("Pilih gambar (JPG/PNG):", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Tampilkan gambar asli
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Gambar Asli", use_container_width=True)

    # Tombol proses
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
                    restored = 1 - img_array[0]  # fallback

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
                href = f'<a href="data:file/jpg;base64,{b64}" download="restored.jpg">⬇️ Download Hasil</a>'
                st.markdown(href, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")
