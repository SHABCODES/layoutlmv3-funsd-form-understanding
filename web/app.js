// Vanilla JS demo client for the FastAPI service in app/api.py.
// No build step on purpose — this should be openable directly as a file:// page
// or served statically, so a recruiter can run it with zero setup beyond the API.

const API_URL = "http://localhost:8000/predict";

const fileInput = document.getElementById("fileInput");
const predictBtn = document.getElementById("predictBtn");
const statusEl = document.getElementById("status");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

let currentImage = null;

fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  if (!file) return;

  const img = new Image();
  img.onload = () => {
    currentImage = img;
    canvas.width = img.width;
    canvas.height = img.height;
    ctx.drawImage(img, 0, 0);
    predictBtn.disabled = false;
    statusEl.textContent = `Loaded ${file.name} (${img.width}\u00d7${img.height})`;
  };
  img.src = URL.createObjectURL(file);
});

predictBtn.addEventListener("click", async () => {
  const file = fileInput.files[0];
  if (!file) return;

  predictBtn.disabled = true;
  statusEl.textContent = "Running OCR + model inference\u2026";

  try {
    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch(API_URL, { method: "POST", body: formData });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Request failed (${res.status})`);
    }
    const data = await res.json();
    drawPredictions(data);
    statusEl.textContent = `Done \u2014 ${data.words.length} words labeled.`;
  } catch (e) {
    statusEl.textContent = `Error: ${e.message}`;
  } finally {
    predictBtn.disabled = false;
  }
});

function drawPredictions(data) {
  // Redraw the base image, then overlay boxes color-coded by entity type.
  ctx.drawImage(currentImage, 0, 0);

  const scaleX = canvas.width / 1000;
  const scaleY = canvas.height / 1000;

  for (const w of data.words) {
    if (w.entity === "O") continue;
    const [x0, y0, x1, y1] = w.box;
    ctx.strokeStyle = w.color;
    ctx.lineWidth = 2;
    ctx.strokeRect(x0 * scaleX, y0 * scaleY, (x1 - x0) * scaleX, (y1 - y0) * scaleY);
  }
}
