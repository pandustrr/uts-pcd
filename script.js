const uploadInput = document.getElementById("upload");
const originalImage = document.getElementById("original-image");
const restoredImage = document.getElementById("restored-image");
const restoreButton = document.getElementById("restore-button");
const resetButton = document.getElementById("reset-button");
const downloadButton = document.getElementById("download-button");
const loadingText = document.getElementById("loading-text");
const originalPlaceholder = document.getElementById("original-placeholder");
const restoredPlaceholder = document.getElementById("restored-placeholder");

// Sembunyikan tombol di awal
restoreButton.classList.add("hidden");
resetButton.classList.add("hidden");
loadingText.classList.add("hidden");
downloadButton.classList.add("hidden");

uploadInput.addEventListener("change", function () {
    const file = this.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = function (e) {
        originalImage.src = e.target.result;
        originalImage.classList.remove("hidden");
        originalPlaceholder.classList.add("hidden");
        restoredImage.classList.add("hidden");
        restoredPlaceholder.classList.remove("hidden");
        downloadButton.classList.add("hidden");
        restoreButton.classList.remove("hidden");
        resetButton.classList.remove("hidden");
    };
    reader.readAsDataURL(file);
});

restoreButton.addEventListener("click", async function () {
    const file = uploadInput.files[0];
    if (!file) {
        alert("Silakan pilih gambar terlebih dahulu!");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    loadingText.classList.remove("hidden");
    restoreButton.disabled = true;
    resetButton.disabled = true;
    downloadButton.classList.add("hidden");

    try {
        const response = await fetch("http://127.0.0.1:5000/restore", {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error("Gagal memproses gambar: " + errorText);
        }

        const data = await response.json();

        if (data.restored_image) {
            // tampilkan hasil
            restoredImage.src = `data:image/jpeg;base64,${data.restored_image}`;
            restoredImage.classList.remove("hidden");
            restoredPlaceholder.classList.add("hidden");
            originalImage.classList.remove("hidden");

            // ✅ aktifkan tombol download
            downloadButton.href = `data:image/jpeg;base64,${data.restored_image}`;
            downloadButton.classList.remove("hidden");
        } else {
            alert("Gagal memproses gambar: " + (data.error || "Tidak diketahui"));
        }
    } catch (error) {
        console.error("Error:", error);
        alert("❌ Tidak dapat terhubung ke server Flask atau terjadi kesalahan.");
    } finally {
        loadingText.classList.add("hidden");
        restoreButton.disabled = false;
        resetButton.disabled = false;
    }
});

resetButton.addEventListener("click", function () {
    uploadInput.value = "";
    originalImage.src = "";
    restoredImage.src = "";
    downloadButton.href = "#";
    originalImage.classList.add("hidden");
    restoredImage.classList.add("hidden");
    originalPlaceholder.classList.remove("hidden");
    restoredPlaceholder.classList.remove("hidden");
    restoreButton.classList.add("hidden");
    resetButton.classList.add("hidden");
    downloadButton.classList.add("hidden");
});