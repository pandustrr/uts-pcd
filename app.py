import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from io import BytesIO
import os

# === Konfigurasi Halaman ===
st.set_page_config(
    page_title="Aplikasi Restorasi Citra",
    page_icon="🧠",
    layout="wide"
)

# === CSS Kustom untuk UI yang lebih rapi ===
st.markdown("""
    <style>
        h1, h2, h3, h4 { font-family: 'Poppins', sans-serif; }
        .stButton>button {
            background-color: #2563eb;
            color: white;
            border: none;
            padding: 0.6rem 1.2rem;
            border-radius: 8px;
            font-weight: 600;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #1d4ed8;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .css-1dp5vir {
            background-color: transparent !important;
        }
    </style>
""", unsafe_allow_html=True)

# === Fungsi untuk memuat model ===
@st.cache_resource
def load_restore_model():
    try:
        MODEL_PATH = "model/model_restorasi_citra.h5"
        
        if not os.path.exists(MODEL_PATH):
            st.error(f"❌ File model tidak ditemukan di: `{MODEL_PATH}`")
            st.info("💡 Pastikan file model ada di folder `model/`")
            return None
        
        model = tf.keras.models.load_model(MODEL_PATH)
        st.toast("✅ Model berhasil dimuat!", icon="✅")
        return model
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {str(e)}")
        return None


# === Load model ===
model = load_restore_model()

# === Header ===
st.title("🧠 Aplikasi Restorasi Citra Digital")
st.markdown("""
Restorasi citra bertujuan untuk **memperbaiki kualitas gambar** yang rusak atau buram.  
Unggah gambar di bawah ini dan sistem akan melakukan pemulihan otomatis.
""")

# === Status Model ===
if model:
    st.success("✅ **Status Model:** Siap digunakan")
else:
    st.warning("⚠️ **Status Model:** Tidak tersedia. Sistem menggunakan pemrosesan dasar.")

st.divider()

# === Upload Gambar ===
st.subheader("📤 Upload Gambar")
uploaded_file = st.file_uploader(
    "Pilih file gambar (JPG, JPEG, PNG):",
    type=["jpg", "jpeg", "png"],
    help="Maksimal ukuran file: 200MB"
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        # Layout dua kolom untuk tampilan sebelum/sesudah
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🖼️ Gambar Asli")
            st.image(image, use_container_width=True)
            st.caption(f"**Ukuran:** {image.size[0]}x{image.size[1]} px | **Mode:** {image.mode}")

        # Tombol proses restorasi
        if st.button("🔧 Proses Restorasi Gambar", type="primary", use_container_width=True):
            with st.spinner("⏳ Memproses gambar..."):
                try:
                    # Preprocessing
                    target_size = (128, 128)
                    img_resized = image.resize(target_size)
                    img_array = np.array(img_resized) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)

                    # Prediksi
                    if model:
                        restored_array = model.predict(img_array, verbose=0)[0]
                        st.toast("✅ Restorasi menggunakan model berhasil!", icon="✨")
                    else:
                        restored_array = img_array[0]
                        st.info("ℹ️ Menggunakan pemrosesan dasar (tanpa model)")

                    # Postprocessing
                    restored_array = (restored_array * 255).astype(np.uint8)
                    restored_img = Image.fromarray(restored_array)
                    restored_display = restored_img.resize(image.size, Image.Resampling.LANCZOS)

                    with col2:
                        st.markdown("### ✨ Hasil Restorasi")
                        st.image(restored_display, use_container_width=True)

                        # Tombol download hasil
                        buffer = BytesIO()
                        restored_display.save(buffer, format="JPEG", quality=95)
                        buffer.seek(0)

                        st.download_button(
                            label="⬇️ Download Hasil Restorasi",
                            data=buffer,
                            file_name="hasil_restorasi.jpg",
                            mime="image/jpeg",
                            use_container_width=True
                        )

                except Exception as e:
                    st.error(f"❌ Error selama proses restorasi: {str(e)}")

    except Exception as e:
        st.error(f"❌ Error memproses gambar: {str(e)}")
