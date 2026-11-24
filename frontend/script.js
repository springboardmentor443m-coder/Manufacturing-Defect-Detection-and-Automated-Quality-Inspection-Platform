const API_URL = "http://127.0.0.1:8000";

const imageInput = document.getElementById("imageInput");
const uploadArea = document.getElementById("uploadArea");
const previewContainer = document.getElementById("previewContainer");
const previewImage = document.getElementById("previewImage");
const fileName = document.getElementById("fileName");
const removeButton = document.getElementById("removeButton");
const inspectButton = document.getElementById("inspectButton");
const feedback = document.getElementById("feedback");

let selectedFile = null;

// Select an image
imageInput.addEventListener("change", function () {
    const file = imageInput.files[0];
    if (file) {
        handleImage(file);
    }
});

// Drag and drop
uploadArea.addEventListener("dragover", function (event) {
    event.preventDefault();
    uploadArea.style.borderColor = "#3478e5";
});

uploadArea.addEventListener("dragleave", function () {
    uploadArea.style.borderColor = "#c8d6e8";
});

uploadArea.addEventListener("drop", function (event) {
    event.preventDefault();
    uploadArea.style.borderColor = "#c8d6e8";

    const file = event.dataTransfer.files[0];

    if (file) {
        handleImage(file);
    }
});

// Display selected image
function handleImage(file) {
    if (!file.type.startsWith("image/")) {
        feedback.textContent = "Please select a valid image.";
        return;
    }

    selectedFile = file;

    const imageURL = URL.createObjectURL(file);
    previewImage.src = imageURL;
    fileName.textContent = file.name;

    uploadArea.hidden = true;
    previewContainer.hidden = false;

    inspectButton.disabled = false;
    feedback.textContent = "Image ready for AI inspection.";
}

// Remove selected image
removeButton.addEventListener("click", function () {
    if (previewImage.src.startsWith("blob:")) {
        URL.revokeObjectURL(previewImage.src);
    }

    selectedFile = null;
    imageInput.value = "";

    previewImage.src = "";
    previewContainer.hidden = true;
    uploadArea.hidden = false;

    inspectButton.disabled = true;
    feedback.textContent = "Upload an image to begin inspection.";

    document.getElementById("emptyResult").hidden = false;
    document.getElementById("resultDetails").hidden = true;
    document.getElementById("visualizationPanel").hidden = true;
});

// Run AI inspection
inspectButton.addEventListener("click", async function () {
    if (!selectedFile) {
        feedback.textContent = "Please upload an image first.";
        return;
    }

    const formData = new FormData();
    formData.append("file", selectedFile);

    inspectButton.disabled = true;
    inspectButton.textContent = "Analyzing image...";
    feedback.textContent = "Sending image to the AI inspection engine...";

    try {
        const response = await fetch(`${API_URL}/inspect`, {
            method: "POST",
            body: formData
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.detail || "Inspection failed.");
        }

        displayResults(result);

        feedback.textContent = "AI inspection completed successfully.";

    } catch (error) {
        console.error("Inspection error:", error);

        feedback.textContent =
            "Error: " + error.message +
            " Check whether the backend is running.";

    } finally {
        inspectButton.disabled = !selectedFile;
        inspectButton.innerHTML = 'Run AI Inspection <span>→</span>';
    }
});

// Display inspection results
function displayResults(result) {
    document.getElementById("emptyResult").hidden = true;
    document.getElementById("resultDetails").hidden = false;

    const status = String(result.status || "REVIEW").toUpperCase();

    const decisionCard = document.getElementById("decisionCard");
    decisionCard.className = "decision-card " + status.toLowerCase();

    document.getElementById("decisionStatus").textContent = status;

    document.getElementById("decisionMessage").textContent =
        status === "PASS"
            ? "No significant anomaly was detected by the current model threshold."
            : "Potential anomalies detected. Further quality review is recommended.";

    document.getElementById("anomalousPatches").textContent =
        result.anomalous_patches ?? "--";

    document.getElementById("totalPatches").textContent =
        result.total_patches ?? "--";

    document.getElementById("anomalyRatio").textContent =
        result.anomaly_ratio !== undefined
            ? (Number(result.anomaly_ratio) * 100).toFixed(2) + "%"
            : "--";

    document.getElementById("decisionThreshold").textContent =
        result.decision_threshold !== undefined
            ? (Number(result.decision_threshold) * 100).toFixed(2) + "%"
            : "--";

    document.getElementById("inspectionId").textContent =
        result.inspection_id || "--";

    // Display generated visualizations
    if (result.visualizations) {
        const original = getImageURL(result.visualizations.original);
        const heatmap = getImageURL(result.visualizations.heatmap);
        const overlay = getImageURL(result.visualizations.overlay);

        document.getElementById("originalImage").src = original;
        document.getElementById("heatmapImage").src = heatmap;
        document.getElementById("overlayImage").src = overlay;

        document.getElementById("visualizationPanel").hidden = false;
    }
}

// Convert backend file paths into browser URLs
function getImageURL(path) {
    const normalizedPath = String(path).replace(/\\/g, "/");

    const artifactPath = normalizedPath.startsWith("artifacts/")
        ? normalizedPath.substring("artifacts/".length)
        : normalizedPath;

    return `${API_URL}/artifacts/${artifactPath}`;
}
