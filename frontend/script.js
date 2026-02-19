let image = null;
let ocrData = [];
let pdfDoc = null;
let currentPage = 1;
let isPDF = false;
let backendWidth = 0;
let backendHeight = 0;
let selectedBox = null;

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const uploadBtn = document.getElementById("uploadBtn");
const ocrWordsBtn = document.getElementById("ocrWordsBtn");
const ocrLinesBtn = document.getElementById("ocrLinesBtn");
const replaceBtn = document.getElementById("replaceBtn");


// =======================
// Upload
// =======================
uploadBtn.addEventListener("click", async function () {

    const file = document.getElementById("fileInput").files[0];

    if (!file) {
        alert("Select a file first");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch("http://127.0.0.1:8000/upload", {
        method: "POST",
        body: formData
    });

    const result = await response.json();
    document.getElementById("savedFilename").value = result.filename;

    if (file.type === "application/pdf") {
        isPDF = true;
        image = null;

        const typedarray = new Uint8Array(await file.arrayBuffer());
        pdfDoc = await pdfjsLib.getDocument(typedarray).promise;

        currentPage = 1;
        renderPDFPage(currentPage);
    } else {
        isPDF = false;
        pdfDoc = null;

        image = new Image();
        image.src = URL.createObjectURL(file);

        image.onload = function () {
            canvas.width = image.width;
            canvas.height = image.height;
            ctx.drawImage(image, 0, 0);
        };
    }
});


// =======================
// Render PDF
// =======================
async function renderPDFPage(pageNumber) {
    const page = await pdfDoc.getPage(pageNumber);
    const viewport = page.getViewport({ scale: 1.5 });

    canvas.width = viewport.width;
    canvas.height = viewport.height;

    await page.render({
        canvasContext: ctx,
        viewport: viewport
    }).promise;
}


// =======================
// OCR
// =======================
async function runOcr(mode) {

    const filename = document.getElementById("savedFilename").value;
    if (!filename) {
        alert("Upload file first!");
        return;
    }

    const response = await fetch(
        `http://127.0.0.1:8000/ocr?filename=${filename}&mode=${mode}`,
        { method: "POST" }
    );

    const result = await response.json();

    if (result.error) {
        alert(result.error);
        return;
    }

    ocrData = result.boxes;
    backendWidth = result.width;
    backendHeight = result.height;

    drawBoxes();
}

ocrWordsBtn.addEventListener("click", () => runOcr("words"));
ocrLinesBtn.addEventListener("click", () => runOcr("lines"));


// =======================
// Draw Boxes
// =======================
function drawBoxes() {

    if (image) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
    }

    ctx.strokeStyle = "red";
    ctx.lineWidth = 2;

    let scaleX = 1, scaleY = 1;

    if (backendWidth > 0 && backendHeight > 0) {
        scaleX = canvas.width / backendWidth;
        scaleY = canvas.height / backendHeight;
    }

    ocrData.forEach(box => {
        ctx.strokeRect(
            box.x * scaleX,
            box.y * scaleY,
            box.width * scaleX,
            box.height * scaleY
        );
    });
}


// =======================
// Select Box On Click
// =======================
canvas.addEventListener("click", function (e) {

    const rect = canvas.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const clickY = e.clientY - rect.top;

    let scaleX = canvas.width / backendWidth;
    let scaleY = canvas.height / backendHeight;

    selectedBox = null;

    for (let box of ocrData) {

        const x = box.x * scaleX;
        const y = box.y * scaleY;
        const w = box.width * scaleX;
        const h = box.height * scaleY;

        if (
            clickX >= x &&
            clickX <= x + w &&
            clickY >= y &&
            clickY <= y + h
        ) {
            selectedBox = box;
            alert("Selected: " + box.text);
            break;
        }
    }
});


// =======================
// Replace Word
// =======================
replaceBtn.addEventListener("click", async function () {

    if (!selectedBox) {
        alert("Select a word or line first (click on a red box)");
        return;
    }

    const filename = document.getElementById("savedFilename").value;

    // Step 1: Detect the complete text in the selected region first
    const detectRes = await fetch("http://127.0.0.1:8000/detect-in-box", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filename: filename, box: selectedBox })
    });
    const detectData = await detectRes.json();
    if (detectData.error) {
        alert("Could not detect text: " + detectData.error);
        return;
    }
    const detectedText = (detectData.detected_text || "").trim() || "(no text detected)";

    // Step 2: Show detected text and ask for replacement
    const newText = prompt(
        "Detected text in selection:\n\n\"" + detectedText + "\"\n\nEnter new text to replace with (same style/weight/height will be applied):",
        detectedText
    );
    if (newText === null) return;
    const newTextTrimmed = (newText || "").trim();
    if (!newTextTrimmed) {
        alert("No new text entered.");
        return;
    }

    const response = await fetch("http://127.0.0.1:8000/replace", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            filename: filename,
            box: selectedBox,
            new_text: newTextTrimmed
        })
    });

    const result = await response.json();

    if (result.error) {
        alert(result.error);
        return;
    }

    image = new Image();
    image.src = result.image_url + "?t=" + new Date().getTime();

    image.onload = function () {
        canvas.width = image.width;
        canvas.height = image.height;
        ctx.drawImage(image, 0, 0);
    };

    alert("Text replaced successfully.");
});