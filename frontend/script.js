async function uploadImage() {
  const fileInput = document.getElementById('fileInput');
  if (!fileInput.files.length) {
    alert("Please select an image first.");
    return;
  }
  const formData = new FormData();
  formData.append("file", fileInput.files[0]);

  const response = await fetch("http://127.0.0.1:5000/predict", {
    method: "POST",
    body: formData
  });

  const result = await response.json();
  document.getElementById("result").innerHTML =
    `<h2>Prediction: ${result.prediction}</h2><p>Confidence: ${(result.confidence*100).toFixed(2)}%</p>`;
}