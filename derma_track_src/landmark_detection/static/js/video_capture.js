async function isCameraConnected(deviceId) {

    let devices = await navigator.mediaDevices.enumerateDevices();

    let cameras = devices.filter(device => device.kind === 'videoinput');

    return cameras.some(camera => camera.deviceId === deviceId);
}

async function stream() {
    let deviceId = "5D9dGdRalU765KCrHr3EhQx0W6LlRWVCzBTGmHTxR/w="
    if (await isCameraConnected(deviceId)){
        let video = document.getElementById("stream");
        video.muted = true;
        navigator.mediaDevices.getUserMedia({video: { deviceId: { exact: deviceId} }})
            .then((stream) => {
                video.srcObject = stream;
            })
            .catch((error) => {
                console.log("Error accessing the camera: ", error);
            })
    }else{
        console.error("The camera of the special device is not connected, please connect it or contact the IT if it's connected but not detected.");
    }
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
