let model = null;

const statusBox = document.getElementById("status");
const uploadForm = document.getElementById("uploadForm");
const imageInput = document.getElementById("imageInput");
const analyzeButton = document.getElementById("analyzeButton");
const resultCard = document.getElementById("resultCard");
const resultLabel = document.getElementById("resultLabel");
const resultMeta = document.getElementById("resultMeta");
const imageWrap = document.getElementById("imageWrap");
const previewImage = document.getElementById("previewImage");

function setStatus(message, mode = "") {
  statusBox.textContent = message;
  statusBox.className = `alert ${mode}`.trim();
}

async function loadModel() {
  try {
    setStatus("Loading model. Please wait...");
    model = await mobilenet.load();
    analyzeButton.disabled = false;
    setStatus("Model ready. Upload an image to classify.", "success");
  } catch (error) {
    setStatus("Could not load model. Check internet and refresh.", "error");
    console.error(error);
  }
}

function mapPredictionToMask(topPrediction) {
  const name = topPrediction.className.toLowerCase();
  const score = topPrediction.probability;
  const maskKeywords = [
    "mask",
    "gasmask",
    "respirator",
    "oxygen mask",
    "surgical",
  ];
  const isMask = maskKeywords.some((keyword) => name.includes(keyword));

  return {
    label: isMask ? "Mask" : "No Mask",
    confidence: score,
    cssClass: isMask ? "mask" : "no-mask",
  };
}

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  if (!model) {
    setStatus("Model is not ready yet.", "error");
    return;
  }

  const file = imageInput.files[0];
  if (!file) {
    setStatus("Please choose an image.", "error");
    return;
  }

  const imageUrl = URL.createObjectURL(file);
  previewImage.src = imageUrl;
  imageWrap.hidden = false;

  const tempImage = new Image();
  tempImage.src = imageUrl;

  await new Promise((resolve, reject) => {
    tempImage.onload = resolve;
    tempImage.onerror = reject;
  });

  try {
    setStatus("Analyzing image...");
    const predictions = await model.classify(tempImage, 3);
    const result = mapPredictionToMask(predictions[0]);

    resultCard.hidden = false;
    resultLabel.textContent = result.label;
    resultLabel.className = result.cssClass;
    resultMeta.textContent = `Confidence: ${(result.confidence * 100).toFixed(2)}%`;
    setStatus("Analysis complete.", "success");
  } catch (error) {
    setStatus("Could not analyze the image.", "error");
    console.error(error);
  }
});

loadModel();
