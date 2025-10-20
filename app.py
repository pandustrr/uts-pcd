import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import base64
from io import BytesIO
import os

# === Konfigurasi Halaman ===
st.set_page_config(
    page_title="Aplikasi Restorasi Citra",
    page_icon="🧠",
    layout="wide"
)

# === Load Model ===
@st.cache_resource
def load_restore_model():
    try:
        # Path model yang benar
        MODEL_PATH = "model/model_restorasi_citra.h5"
        
        if not os.path.exists(MODEL_PATH):
            st.error(f"❌ File model tidak ditemukan di: {MODEL_PATH}")
            st.info("💡 Pastikan file model_restorasi_citra.h5 ada di folder 'model/'")
            return None
        
        model = tf.keras.models.load_model(MODEL_PATH)
        st.success("✅ Model berhasil dimuat!")
        return model
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {str(e)}")
        return None

# Load model
model = load_restore_model()

# === UI Streamlit ===
st.title("🧠 Aplikasi Restorasi Citra Digital")
st.markdown("Unggah gambar yang ingin direstorasi, dan sistem AI akan memperbaikinya.")

# Informasi status model
if model:
    st.success("✅ **Status:** Model AI siap digunakan")
else:
    st.warning("⚠️ **Status:** Model tidak tersedia. Gunakan gambar dengan kualitas baik.")

# Upload section
st.subheader("📤 Upload Gambar")
uploaded_file = st.file_uploader(
    "Pilih file gambar (JPG, JPEG, PNG):",
    type=["jpg", "jpeg", "png"],
    help="Maksimal ukuran file: 200MB"
)

if uploaded_file is not None:
    # Process the image
    try:
        image = Image.open(uploaded_file).convert("RGB")
        
        # Display original image
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🖼️ Gambar Asli")
            st.image(image, use_column_width=True)  # PERBAIKAN: ganti use_container_width
            st.write(f"**Ukuran:** {image.size[0]} x {image.size[1]} pixels")
            st.write(f"**Mode:** {image.mode}")

        # Restoration button
        if st.button("🔧 PROSES RESTORASI", type="primary", use_column_width=True):
            with st.spinner("🔄 Sedang memproses gambar... Mohon tunggu"):
                try:
                    # Preprocessing
                    target_size = (128, 128)  # Sesuai dengan input model
                    img_resized = image.resize(target_size)
                    img_array = np.array(img_resized) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)

                    # Prediction
                    if model:
                        restored_array = model.predict(img_array, verbose=0)[0]
                    else:
                        # Fallback: return original (dengan beberapa enhancement sederhana)
                        restored_array = img_array[0]

                    # Postprocessing
                    restored_array = (restored_array * 255).astype(np.uint8)
                    restored_img = Image.fromarray(restored_array)
                    
                    # Resize back to original dimensions for display
                    restored_display = restored_img.resize(image.size, Image.Resampling.LANCZOS)

                    with col2:
                        st.subheader("✨ Hasil Restorasi")
                        st.image(restored_display, use_column_width=True)  # PERBAIKAN: ganti use_container_width
                        st.success("✅ Restorasi selesai!")
                        
                        # Download button
                        buffer = BytesIO()
                        restored_display.save(buffer, format="JPEG", quality=95)
                        buffer.seek(0)
                        
                        st.download_button(
                            label="⬇️ DOWNLOAD HASIL",
                            data=buffer,
                            file_name="hasil_restorasi.jpg",
                            mime="image/jpeg",
                            type="primary",
                            use_column_width=True
                        )

                except Exception as e:
                    st.error(f"❌ Error selama proses restorasi: {str(e)}")
                    
    except Exception as e:
        st.error(f"❌ Error memproses gambar: {str(e)}")

