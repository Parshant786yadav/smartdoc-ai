let image = null;
let ocrData = [];

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const uploadBtn = document.getElementById("uploadBtn");
const ocrBtn = document.getElementById("ocrBtn");

uploadBtn.addEventListener("click", async function () {

    const fileInput = document.getElementById("fileInput");
    const file = fileInput.files[0];

    if (!file) {
        alert("Select a file first");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
        const response = await fetch("http://127.0.0.1:8000/upload", {
            method: "POST",
            body: formData
        });

        const result = await response.json();

        if (!result.filename) {
            alert("Upload failed");
            console.log(result);
            return;
        }

        document.getElementById("savedFilename").value = result.filename;

        image = new Image();
        image.src = URL.createObjectURL(file);

        image.onload = function () {
            canvas.width = image.width;
            canvas.height = image.height;
            ctx.drawImage(image, 0, 0);
        };

    } catch (error) {
        console.error("Upload error:", error);
        alert("Upload failed");
    }
});


ocrBtn.addEventListener("click", async function () {

    const filename = document.getElementById("savedFilename").value;

    if (!filename) {
        alert("Upload file first!");
        return;
    }

    try {
        const response = await fetch(
            `http://127.0.0.1:8000/ocr?filename=${encodeURIComponent(filename)}`,
            { method: "POST" }
        );

        const result = await response.json();

        if (!result.data) {
            alert("OCR failed");
            console.log(result);
            return;
        }

        ocrData = result.data;

        drawBoxes();

    } catch (error) {
        console.error("OCR error:", error);
        alert("OCR request failed");
    }
});


function drawBoxes() {

    if (!image) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(image, 0, 0);

    ctx.strokeStyle = "red";
    ctx.lineWidth = 2;

    // Calculate scaling ratio
    let scaleX = canvas.width / image.naturalWidth;
    let scaleY = canvas.height / image.naturalHeight;

    ocrData.forEach(box => {
        ctx.strokeRect(
            box.x * scaleX,
            box.y * scaleY,
            box.width * scaleX,
            box.height * scaleY
        );
    });
}

