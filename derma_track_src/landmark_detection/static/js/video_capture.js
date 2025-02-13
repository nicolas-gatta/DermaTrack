const video = document.createElement("video");
video.setAttribute("autoplay", "");
video.setAttribute("playsinline", "");
document.body.appendChild(video);

const canvas = document.createElement("canvas");
document.body.appendChild(canvas);
const ctx = canvas.getContext("2d");

navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        video.srcObject = stream;
    })
    .catch(err => console.error("Error accessing camera: ", err));

function captureFrame() {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
}

setInterval(captureFrame, 100);