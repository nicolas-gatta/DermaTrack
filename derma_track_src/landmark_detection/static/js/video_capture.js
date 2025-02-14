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

document.getElementById("start").addEventListener("click", stream);