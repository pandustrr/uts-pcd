import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# === FUNGSI LOAD MODEL ===
@st.cache_resource
def load_restore_model():
    try:
        model = tf.keras.models.load_model("model_cnn.h5")
        return model
    except Exception as e:
        # Jangan panggil st.error di dalam fungsi cache!
        return str(e)

model_result = load_restore_model()
if isinstance(model_result, str):
    st.error(f"❌ Gagal memuat model: {model_result}")
    st.stop()
else:
    model = model_result

# === CUSTOM HEADER ===
st.markdown("""
    <style>
        .main-title {
            text-align: center;
            color: white;
            background-color: #1E3A8A;
            padding: 12px;
            border-radius: 12px;
            font-size: 28px;
            font-weight: bold;
        }
        .result-box {
            background-color: #F3F4F6;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🔍 Klasifikasi Citra Daun Menggunakan CNN</div>', unsafe_allow_html=True)
st.write("Unggah gambar daun untuk memprediksi jenisnya.")

# === INPUT GAMBAR ===
uploaded_file = st.file_uploader("📂 Upload gambar daun", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar diunggah", width=300)

    img_array = image.resize((128, 128))
    img_array = np.array(img_array) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediksi
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]

    class_names = ["Daun Jambu", "Daun Pepaya", "Daun Singkong"]

    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    st.subheader("🌿 Hasil Prediksi")
    st.write(f"**Jenis Daun:** {class_names[predicted_class]}")
    st.write(f"**Probabilitas:** {np.max(prediction) * 100:.2f}%")
    st.markdown("</div>", unsafe_allow_html=True)
else:
    st.info("Silakan unggah gambar terlebih dahulu.")

# === Footer ===
st.markdown("""
    <hr>
    <div style='text-align:center; color:gray'>
        <small>Dibuat untuk UTS Pengolahan Citra Digital 🌱</small>
    </div>
""", unsafe_allow_html=True)
