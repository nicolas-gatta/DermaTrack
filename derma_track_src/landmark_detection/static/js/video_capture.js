async function getCameraDeviceIdByName(cameraName) {
    await navigator.mediaDevices.getUserMedia({ video: true });
    const devices = await navigator.mediaDevices.enumerateDevices();
    const cameras = devices.filter(device => device.kind === 'videoinput');
    const targetCamera = cameras.find(camera => camera.label.includes(cameraName));
    return targetCamera ? targetCamera.deviceId : null;
}


async function isCameraConnected(deviceId) {

    let devices = await navigator.mediaDevices.enumerateDevices();

    let cameras = devices.filter(device => device.kind === 'videoinput');

    return cameras.some(camera => camera.deviceId === deviceId);
}

function populateSelectBodyPart(){
    fetch(`/landmark/get_body_parts/`, {
        method: "GET",
        headers: { 
            "Content-Type": "application/json"
        }
    })
    .then(response => response.json())
    .then(data => {
        const selector = document.getElementById('body-part');
        selector.innerHTML = '<option value="" disabled selected>Select a body part...</option>'; 

        data.forEach((bodyPart) => {
            const option = document.createElement('option');
            option.value = bodyPart[0];
            option.textContent = bodyPart[1];
            selector.appendChild(option);
        });
    })
}

async function stream() {
    populateSelectBodyPart();
    localStorage.setItem("capturedImages", JSON.stringify([]));
    localStorage.setItem("bodyPartImages", JSON.stringify([]));
    updateCarousel([]);
    let deviceId = await getCameraDeviceIdByName("Arducam IMX179 8MP Camera");
    if (await isCameraConnected(deviceId)){
        let video = document.getElementById("stream");
        video.muted = true;
        navigator.mediaDevices.getUserMedia({video: true, width: { ideal:  1920} , height: { ideal: 1080 }},)
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
    let bodyPart = document.getElementById("body-part").selectedOptions[0].value;

    context.drawImage(video, 0, 0, video.videoWidth, video.videoHeight);

    imageUrl = canvas.toDataURL("image/png")
    if (isSaved){
        saveImageToCache(imageUrl, bodyPart);
    }else{
        return imageUrl.split(",")[1];
    }
}


function saveImageToCache(imageUrl, bodyPart) {
    let storedImages = JSON.parse(localStorage.getItem("capturedImages"));
    let bodyPartImages = JSON.parse(localStorage.getItem("bodyPartImages"));

    storedImages.push(imageUrl);
    bodyPartImages.push(bodyPart);

    localStorage.setItem("capturedImages", JSON.stringify(storedImages));
    localStorage.setItem("bodyPartImages", JSON.stringify(bodyPartImages));

    updateCarousel(storedImages);
}

function saveImagesToServer(visitId) {
    const storedImages = JSON.parse(localStorage.getItem("capturedImages"));
    const bodyPartImages = JSON.parse(localStorage.getItem("bodyPartImages"));

    if (!storedImages || storedImages.length === 0) {
        alert("No images to save!");
        return;
    }

    if (!bodyPartImages || bodyPartImages.length === 0) {
        alert("No Body Part Save!");
        return;
    }

    alert('Image saving process initiated');
    
    storedImages.forEach(async (imageUrl, index) => {
        try {
            fetch(`/landmark/save_images/`, {
                method: "POST",
                headers: { 
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ visitId: visitId, image: imageUrl, bodyPartId: bodyPartImages[index]})
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'error') {
                    console.error(`Failed to save image ${index + 1}: ${data.message}`);
                }
            })

        } catch (error) {
            console.error(`Error saving image ${index + 1}:`, error);
        }
    });
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
