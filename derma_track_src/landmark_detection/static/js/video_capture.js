async function isCameraConnected(deviceId) {

    let devices = await navigator.mediaDevices.enumerateDevices();

    let cameras = devices.filter(device => device.kind === 'videoinput');

    return cameras.some(camera => camera.deviceId === deviceId);
}

async function stream() {
    localStorage.setItem("capturedImages", JSON.stringify([]));
    updateCarousel([]);
    let deviceId = "5D9dGdRalU765KCrHr3EhQx0W6LlRWVCzBTGmHTxR/w="
    if (true || await isCameraConnected(deviceId)){
        let video = document.getElementById("stream");
        video.muted = true;
        //navigator.mediaDevices.getUserMedia({video: { deviceId: { exact: deviceId} }})
        navigator.mediaDevices.getUserMedia({video: true, width: { ideal:  1920} , height: { ideal: 1080 }},)
            .then((stream) => {
                video.srcObject = stream;
                console.log(video);
            })
            .catch((error) => {
                console.log("Error accessing the camera: ", error);
            })
        detectBodyPart();
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

function detectBodyPart(){
    let body_part = null;
    let detection_text = document.getElementById("body");
    let countdown = document.getElementById("countdown");
    let count = 5;

    function sendDetectionRequest(){
        countdown.innerHTML = `${count} secondes` 
        fetch("/landmark/detect", {
            method: "POST",
            headers: { 
                "Content-Type": "application/json",
                "X-CSRFToken": csrftoken
            },
            body: JSON.stringify({ image: captureImage(false) })
        })
        .then(response => response.json())
        .then(data => {
            detection_text.innerText = data.body_part;
            if(body_part != "Unknown"){
                if (data.body_part != body_part){
                    count = 5;
                    body_part = data.body_part;
                }else{
                    count -= 1;
                }
            }
        })

        if(count <= 0){
            clearInterval(interval);
        }
    }

    //interval = setInterval(sendDetectionRequest, 1000)
}

function captureImage(isSaved){
    let video = document.getElementById("stream");
    let canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    let context = canvas.getContext("2d");

    context.drawImage(video, 0, 0, video.videoWidth, video.videoHeight);

    imageUrl = canvas.toDataURL("image/png")
    if (isSaved){
        saveImageToCache(imageUrl);
    }else{
        return imageUrl.split(",")[1];
    }
}

function saveImageToCache(imageUrl) {
    let storedImages = JSON.parse(localStorage.getItem("capturedImages"));

    storedImages.push(imageUrl);

    localStorage.setItem("capturedImages", JSON.stringify(storedImages));

    updateCarousel(storedImages);
}

function updateCarousel(images) {
    let carouselInner = document.querySelector("#medicalCarousel .carousel-inner");
    carouselInner.innerHTML = ""; 

    let itemsPerSlide = 5;
    let totalSlides = Math.ceil(images.length / itemsPerSlide);

    for (let i = 0; i < totalSlides; i++) {
        let carouselItem = document.createElement("div");
        carouselItem.classList.add("carousel-item");

        if (i === 0) carouselItem.classList.add("active");

        let container = document.createElement("div");
        container.classList.add("container-fluid");
        container.style.backgroundColor = "beige";

        let row = document.createElement("div");
        row.classList.add("row", "justify-content-center");

        for (let j = i * itemsPerSlide; j < (i + 1) * itemsPerSlide && j < images.length; j++) {
            let img = document.createElement("img");
            img.classList.add("col-sm-2");
            img.src = images[j];
            img.alt = "Captured Image";
            row.appendChild(img);
        }

        container.appendChild(row);
        carouselItem.appendChild(container);
        carouselInner.appendChild(carouselItem);
    }
}
