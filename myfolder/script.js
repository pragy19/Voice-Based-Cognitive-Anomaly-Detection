async function uploadAudio() {
    const input = document.getElementById("audioInput");
    const file = input.files[0];
    const resultBox = document.getElementById("result");
  
    if (!file) {
      resultBox.textContent = "Please select an audio file first.";
      return;
    }
  
    const formData = new FormData();
    formData.append("file", file);
  
    resultBox.textContent = "Uploading and processing...";
  
    try {
      const response = await fetch("https://voice-based-cognitive-anomaly-detection.onrender.com/predict", {
        method: "POST",
        body: formData
      });
  
      const data = await response.json();
      resultBox.textContent = JSON.stringify(data, null, 2);
    } catch (err) {
      resultBox.textContent = "Error: " + err.message;
    }
  }
  