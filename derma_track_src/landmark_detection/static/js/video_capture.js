function stream() {
    const video = document.getElementById("stream");
    video.muted = true;
    navigator.mediaDevices.getUserMedia({video: true})
        .then((stream) => {
            video.srcObject = stream;
        })
        .catch((error) => {
            console.log("Error accessing the camera: ", error);
        })
}

function stopStream() {
    const video = document.getElementById("stream");
    if (video.srcObject) {
        let tracks = video.srcObject.getTracks();
        tracks.forEach(track => track.stop());
        video.srcObject = null;
    }
}

function captureImage(){
    let video = document.getElementById("stream");
    let canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    let context = canvas.getContext("2d");
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    let imageUrl = canvas.toDataURL("image/png"); // Convert image to base64
    saveImageToCache(imageUrl);
}

function saveImageToCache(imageUrl) {
    let images = JSON.parse(localStorage.getItem("capturedImages")) || [];
    images.push(imageUrl);
    localStorage.setItem("capturedImages", JSON.stringify(images)); // Save images in cache
}
