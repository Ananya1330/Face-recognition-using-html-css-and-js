<!DOCTYPE html>
<html lang="en">
<head>
  <title>Face Recognition</title>
  <script src="https://cdn.jsdelivr.net/npm/face-api.js"></script>
  <style>
    video, canvas, img {
      display: block;
      margin: 10px auto;
    }
  </style>
</head>
<body>

<video id="video" width="640" height="480" autoplay></video>
<button id="capture">Capture Photo</button>
<canvas id="captureCanvas" width="640" height="480" style="display: none;"></canvas>
<img id="capturedImage" src="" alt="Captured Photo" style="display: none;"/>
<div id="result"></div>

<script>
  const video = document.getElementById("video");
  const captureButton = document.getElementById("capture");
  const captureCanvas = document.getElementById("captureCanvas");
  const capturedImage = document.getElementById("capturedImage");
  const resultDiv = document.getElementById("result");

  // Load face-api models and handle errors
  Promise.all([
    faceapi.nets.ssdMobilenetv1.loadFromUri("/models"),
    faceapi.nets.faceRecognitionNet.loadFromUri("/models"),
    faceapi.nets.faceLandmark68Net.loadFromUri("/models"),
  ])
  .then(startWebcam)
  .catch(error => {
    console.error("Error loading models:", error);
    alert("Failed to load models.");
  });

  function startWebcam() {
    navigator.mediaDevices.getUserMedia({ video: true, audio: false })
      .then(stream => {
        video.srcObject = stream;
      })
      .catch(error => {
        console.error("Error accessing webcam:", error);
        alert("Failed to access webcam. Please check permissions.");
      });
  }

  captureButton.addEventListener("click", async () => {
    // Capture the current frame from the video
    const ctx = captureCanvas.getContext("2d");
    ctx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);

    // Get the image as a Data URL and set it to the capturedImage element
    const dataUrl = captureCanvas.toDataURL("image/png");
    capturedImage.src = dataUrl;

    // Display the captured image
    capturedImage.style.display = "block";

    // Perform face detection and recognition
    try {
      const labeledFaceDescriptors = await getLabeledFaceDescriptions();
      const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors);

      const detections = await faceapi.detectSingleFace(capturedImage)
        .withFaceLandmarks()
        .withFaceDescriptors();

      if (detections === null) {
        resultDiv.textContent = "No face detected. Please try again.";
        return;
      }

      const bestMatch = faceMatcher.findBestMatch(detections.descriptor);

      if (bestMatch.label === "unknown") {
        resultDiv.textContent = "User doesn't exist.";
      } else {
        resultDiv.textContent = `Welcome ${bestMatch.label}`;
      }
    } catch (error) {
      console.error("Error in face detection/recognition:", error);
      resultDiv.textContent = "Error in face detection/recognition. Please try again.";
    }
  });

  async function getLabeledFaceDescriptions() {
    const labels = ["ANANYA", "BHARTI", "CHEESE", "DHONI", "GIBBA", "KASHYAP", "NARAVANE", "PONAPPA", "SAM", "SINDHU", "VIRAT"];
    const labeledDescriptors = [];

    for (const label of labels) {
      const descriptions = [];
      for (let i = 1; i <= 12; i++) {
        try {
          const img = await faceapi.fetchImage(`./labels/${label}/${i}.png`);
          const detections = await faceapi.detectSingleFace(img)
            .withFaceLandmarks()
            .withFaceDescriptor();

          if (detections) {
            descriptions.push(detections.descriptor);
          } else {
            console.warn(`No face detected in ${label}/${i}.png`);
          }
        } catch (error) {
          console.error(`Error processing ${label}/${i}.png:`, error);
        }
      }

      if (descriptions.length > 0) {
        labeledDescriptors.push(new faceapi.LabeledFaceDescriptors(label, descriptions));
      }
    }

    return labeledDescriptors;
  }

</script>

</body>
</html>
