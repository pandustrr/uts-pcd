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
    layout="wide",
    initial_sidebar_state="collapsed"
)

# === Custom CSS untuk UI yang lebih menarik ===
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        padding: 2rem;
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .header-title {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-align: center;
    }
    
    .header-subtitle {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
        text-align: center;
        margin-top: 0.5rem;
    }
    
    /* Card styling */
    .status-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
    }
    
    /* Upload section styling */
    .upload-section {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 12px;
        border: 2px dashed #cbd5e0;
        margin: 1.5rem 0;
        text-align: center;
    }
    
    /* Image container */
    .image-container {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Info boxes */
    .info-box {
        background: #e6f3ff;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #3182ce;
        margin: 1rem 0;
    }
    
    /* Metric styling */
    .metric-container {
        background: #f7fafc;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    
    /* Download button */
    .stDownloadButton>button {
        background: #48bb78;
        color: white;
        width: 100%;
        border-radius: 8px;
        height: 2.8rem;
        font-weight: 600;
    }
    
    .stDownloadButton>button:hover {
        background: #38a169;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #667eea !important;
    }
    </style>
""", unsafe_allow_html=True)

# === Load Model ===
@st.cache_resource
def load_restore_model():
    try:
        MODEL_PATH = "model/model_restorasi_citra.h5"
        
        if not os.path.exists(MODEL_PATH):
            st.error(f"❌ File model tidak ditemukan di: {MODEL_PATH}")
            st.info("💡 Pastikan file model_restorasi_citra.h5 ada di folder 'model/'")
            return None
        
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {str(e)}")
        return None

# Load model
model = load_restore_model()

# === Header Section ===
st.markdown("""
    <div class="header-container">
        <h1 class="header-title">🧠 Aplikasi Restorasi Citra Digital</h1>
        <p class="header-subtitle">Teknologi AI untuk memulihkan dan meningkatkan kualitas gambar Anda</p>
    </div>
""", unsafe_allow_html=True)

# === Status Model ===
col_status1, col_status2, col_status3 = st.columns([1, 2, 1])
with col_status2:
    if model:
        st.success("✅ **Model Aktif** | Siap memproses gambar Anda", icon="✅")
    else:
        st.warning("⚠️ **Mode Fallback** | Model tidak tersedia, menggunakan processing dasar", icon="⚠️")

st.markdown("<br>", unsafe_allow_html=True)

# === Upload Section ===
st.markdown("### 📤 Upload Gambar")
st.markdown("Pilih gambar yang ingin Anda restorasi dengan teknologi AI")

uploaded_file = st.file_uploader(
    "Drag and drop atau klik untuk memilih file",
    type=["jpg", "jpeg", "png"],
    help="Format yang didukung: JPG, JPEG, PNG | Maksimal: 200MB",
    label_visibility="collapsed"
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        
        # Separator
        st.markdown("---")
        st.markdown("### 🖼️ Perbandingan Gambar")
        
        # Display images side by side
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            st.markdown("#### 📥 Gambar Asli")
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(image, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Image info
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            info_col1, info_col2 = st.columns(2)
            with info_col1:
                st.metric("Dimensi", f"{image.size[0]} × {image.size[1]} px")
            with info_col2:
                st.metric("Mode Warna", image.mode)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown("#### ✨ Hasil Restorasi")
            result_placeholder = st.empty()
            with result_placeholder.container():
                st.info("👈 Klik tombol 'PROSES RESTORASI' untuk memulai", icon="ℹ️")
        
        # Process button
        st.markdown("<br>", unsafe_allow_html=True)
        
        button_col1, button_col2, button_col3 = st.columns([1, 2, 1])
        with button_col2:
            if st.button("🔧 PROSES RESTORASI", type="primary", use_container_width=True):
                with st.spinner("🔄 Sedang memproses gambar ... Mohon tunggu"):
                    try:
                        # Preprocessing
                        target_size = (128, 128)
                        img_resized = image.resize(target_size)
                        img_array = np.array(img_resized) / 255.0
                        img_array = np.expand_dims(img_array, axis=0)

                        # Prediction
                        if model:
                            restored_array = model.predict(img_array, verbose=0)[0]
                            st.success("✅ Restorasi berhasil menggunakan model AI!", icon="✅")
                        else:
                            restored_array = img_array[0]
                            st.info("ℹ️ Menggunakan processing dasar (model tidak tersedia)", icon="ℹ️")

                        # Postprocessing
                        restored_array = (restored_array * 255).astype(np.uint8)
                        restored_img = Image.fromarray(restored_array)
                        restored_display = restored_img.resize(image.size, Image.Resampling.LANCZOS)

                        # Update result
                        with result_placeholder.container():
                            st.markdown('<div class="image-container">', unsafe_allow_html=True)
                            st.image(restored_display, use_container_width=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Download section
                            st.markdown("<br>", unsafe_allow_html=True)
                            buffer = BytesIO()
                            restored_display.save(buffer, format="JPEG", quality=95)
                            buffer.seek(0)
                            
                            st.download_button(
                                label="⬇️ DOWNLOAD HASIL RESTORASI",
                                data=buffer,
                                file_name=f"restored_{uploaded_file.name}",
                                mime="image/jpeg",
                                use_container_width=True
                            )

                    except Exception as e:
                        st.error(f"❌ Error selama proses restorasi: {str(e)}", icon="❌")
                        
    except Exception as e:
        st.error(f"❌ Error memproses gambar: {str(e)}", icon="❌")

else:
    # Empty state
    st.markdown("<br><br>", unsafe_allow_html=True)
    empty_col1, empty_col2, empty_col3 = st.columns([1, 2, 1])
    with empty_col2:
        st.info("📁 Belum ada gambar yang diupload. Pilih file gambar untuk memulai.", icon="📁")

