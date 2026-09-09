const API_URL = "http://localhost:8000/predict";

const dropZone = document.getElementById("dropZone");
const fileInput = document.getElementById("fileInput");
const predictBtn = document.getElementById("predictBtn");
const statusEl = document.querySelector(".status-text");
const resultsContainer = document.getElementById("resultsContainer");
const canvasContainer = document.getElementById("canvasContainer");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const kvPairsContainer = document.getElementById("kvPairsContainer");
const exportBtn = document.getElementById("exportBtn");

const btnText = document.querySelector(".btn-text");
const loader = document.querySelector(".loader");

let currentImage = null;
let currentFile = null;

// Hex to RGBA helper for semi-transparent fills
function hexToRgba(hex, alpha) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

// Map entity to standard colors to match UI variables
const entityColors = {
  "B-HEADER": "#3b82f6",
  "I-HEADER": "#3b82f6",
  "B-QUESTION": "#ef4444",
  "I-QUESTION": "#ef4444",
  "B-ANSWER": "#22c55e",
  "I-ANSWER": "#22c55e",
};

// Handle Drag & Drop
dropZone.addEventListener("click", () => fileInput.click());

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("dragover");
});

dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("dragover");
});

dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("dragover");
  
  if (e.dataTransfer.files && e.dataTransfer.files[0]) {
    handleFile(e.dataTransfer.files[0]);
  }
});

fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) {
    handleFile(fileInput.files[0]);
  }
});

function handleFile(file) {
  if (!file.type.startsWith("image/")) {
    statusEl.textContent = "Please upload a valid image file.";
    return;
  }
  
  currentFile = file;
  const img = new Image();
  img.onload = () => {
    currentImage = img;
    canvas.width = img.width;
    canvas.height = img.height;
    ctx.drawImage(img, 0, 0);
    
    predictBtn.disabled = false;
    statusEl.textContent = `Loaded ${file.name} (${img.width} \u00d7 ${img.height})`;
    
    // Show results container
    resultsContainer.classList.remove("hidden");
    kvPairsContainer.innerHTML = ""; // Clear old results
  };
  img.src = URL.createObjectURL(file);
}

// Handle Prediction
predictBtn.addEventListener("click", async () => {
  if (!currentFile) return;

  // Set loading state
  predictBtn.disabled = true;
  btnText.textContent = "Analyzing...";
  loader.classList.remove("hidden");
  statusEl.textContent = "Running OCR & neural network inference...";

  try {
    const formData = new FormData();
    formData.append("file", currentFile);

    const res = await fetch(API_URL, { method: "POST", body: formData });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Request failed (${res.status})`);
    }
    
    const data = await res.json();
    drawPredictions(data);
    renderVerificationPanel(data);
    statusEl.textContent = `Success \u2014 mapped ${data.words.length} tokens.`;
  } catch (e) {
    statusEl.textContent = `Error: ${e.message}`;
  } finally {
    // Reset loading state
    predictBtn.disabled = false;
    btnText.textContent = "Run Analysis";
    loader.classList.add("hidden");
  }
});

function drawPredictions(data) {
  // Clear and redraw base image
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(currentImage, 0, 0);

  const scaleX = canvas.width / 1000;
  const scaleY = canvas.height / 1000;

  for (const w of data.words) {
    if (w.entity === "O") continue;
    
    const [x0, y0, x1, y1] = w.box;
    const baseColor = entityColors[w.entity] || w.color;
    
    // Draw semi-transparent fill
    ctx.fillStyle = hexToRgba(baseColor, 0.2);
    ctx.fillRect(x0 * scaleX, y0 * scaleY, (x1 - x0) * scaleX, (y1 - y0) * scaleY);
    
    // Draw solid stroke
    ctx.strokeStyle = baseColor;
    ctx.lineWidth = 2;
    ctx.strokeRect(x0 * scaleX, y0 * scaleY, (x1 - x0) * scaleX, (y1 - y0) * scaleY);
  }
}

function renderVerificationPanel(data) {
  kvPairsContainer.innerHTML = "";
  
  if (!data.structured_data || data.structured_data.length === 0) {
    kvPairsContainer.innerHTML = "<p>No structured key-value pairs detected.</p>";
    return;
  }
  
  data.structured_data.forEach((pair, index) => {
    const confClass = pair.confidence > 0.8 ? "high" : "low";
    const confPercent = Math.round(pair.confidence * 100);
    
    const pairHtml = `
      <div class="kv-pair" data-index="${index}">
        <label>
          Pair #${index + 1}
          <span class="conf ${confClass}">${confPercent}% Conf</span>
        </label>
        <input type="text" class="question-input" value="${pair.question}" placeholder="Question / Key">
        <input type="text" class="answer-input" value="${pair.answer}" placeholder="Answer / Value">
      </div>
    `;
    kvPairsContainer.insertAdjacentHTML("beforeend", pairHtml);
  });
}

exportBtn.addEventListener("click", () => {
  const pairs = [];
  const pairEls = document.querySelectorAll(".kv-pair");
  
  pairEls.forEach((el) => {
    const question = el.querySelector(".question-input").value;
    const answer = el.querySelector(".answer-input").value;
    pairs.push({ question, answer });
  });
  
  const blob = new Blob([JSON.stringify(pairs, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  
  const a = document.createElement("a");
  a.href = url;
  a.download = "verified_results.json";
  a.click();
  URL.revokeObjectURL(url);
});
