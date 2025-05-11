let port;
let reader;
let dataReading = false;
let writer;

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
    localStorage.setItem("distanceImages", JSON.stringify([]));
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
        dataReading = true;
        connectToSerialPort();
    }else{
        console.error("The camera of the special device is not connected, please connect it or contact the IT if it's connected but not detected.");
    }
}

async function connectToSerialPort(){
    try {
        port = await navigator.serial.requestPort();
        await port.open({ baudRate: 115200 }); 
        console.log("Serial port connected.");
        const textDecoder = new TextDecoderStream();
        const textEncoder = new TextEncoderStream();
        const readableStreamClosed = port.readable.pipeTo(textDecoder.writable);
        const writableStreamClosed = textEncoder.readable.pipeTo(port.writable);
        writer = textEncoder.writable.getWriter();
        reader = textDecoder.readable.getReader();
        listenToSerialPort();
    } catch (error) {
        console.error("Failed to connect to the serial port:", error);
    }
}

async function sendDataToSerialPort(data) {

    if (!writer) {
        console.error("Serial port writer is not initialized");
        return;
    }

    try {
        const jsonData = JSON.stringify(data);
        await writer.write(jsonData);
    } catch (error) {
        console.error("Failed to send data:", error);
    }
}

async function disconnectedToSerialPort(){

    dataReading = false;

    if (reader) {
        reader.releaseLock();
        await reader.cancel();
    }
    if (port) {
        await port.close();
        console.log("Serial port closed.");
    }
}

async function listenToSerialPort(){

    console.log("Listening for serial data...");

    buffer = "";

    while (dataReading) {
        const { value, done } = await reader.read();

        if (done || !dataReading) {
            reader.releaseLock();
            break;
        }

        if (value) {

            buffer += value;
            
            let lines = buffer.split("\n");

            if(lines.length > 1){
                lines = lines.filter(line => line && line.trim() !== "");
                if (lines.length !=0) {
                    try{
                        data = JSON.parse(lines[0]);
                        if (data["take_picture"]){
                            sendDataToSerialPort({picture_taken: true})
                            captureImage(true, data["distance"]);
                        }
                    }catch{
                        console.log(lines[0])
                    }
                }
                
                buffer = lines.length >= 2 ? lines.pop() : "";
            }
        }
    }
}

function stopStream() {
    const video = document.getElementById("stream");
    if (video.srcObject) {
        let tracks = video.srcObject.getTracks();
        tracks.forEach(track => track.stop());
        video.srcObject = null;
    }

    disconnectedToSerialPort();
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

async function captureImage(isSaved, distance = null){
    let video = document.getElementById("stream");
    let canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    let context = canvas.getContext("2d");
    let bodyPart = document.getElementById("body-part").selectedOptions[0].value;

    context.drawImage(video, 0, 0, video.videoWidth, video.videoHeight);

    imageUrl = canvas.toDataURL("image/png")
    if (isSaved){
        await saveImageToCache(imageUrl, bodyPart, distance);
    }else{
        return imageUrl.split(",")[1];
    }
}

function saveImageToCache(imageUrl, bodyPart, distance) {
    let storedImages = JSON.parse(localStorage.getItem("capturedImages"));
    let bodyPartImages = JSON.parse(localStorage.getItem("bodyPartImages"));
    let distanceImages = JSON.parse(localStorage.getItem("distanceImages"));

    storedImages.push(imageUrl);
    bodyPartImages.push(bodyPart);
    distanceImages.push(distance);

    localStorage.setItem("capturedImages", JSON.stringify(storedImages));
    localStorage.setItem("bodyPartImages", JSON.stringify(bodyPartImages));
    localStorage.setItem("distanceImages", JSON.stringify(distanceImages));

    updateCarousel(storedImages);
}

async function saveImagesToServer(visitId) {
    const storedImages = JSON.parse(localStorage.getItem("capturedImages"));
    const bodyPartImages = JSON.parse(localStorage.getItem("bodyPartImages"));
    const distanceImages = JSON.parse(localStorage.getItem("distanceImages"));

    if (!storedImages || storedImages.length === 0) {
        alert("No images to Save !");
        return;
    }

    if (!bodyPartImages || bodyPartImages.length === 0) {
        alert("No Body Part Save !");
        return;
    }

    if (!distanceImages || distanceImages.length === 0) {
        alert("No Distance Save !");
        return;
    }

    alert('Image saving process initiated');
    
    storedImages.forEach(async (imageUrl, index) => {
        try {
            var response = await fetch(`/landmark/save_images/`, {
                method: "POST",
                headers: { 
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ visitId: visitId, image: imageUrl, bodyPartId: bodyPartImages[index], distance: distanceImages[index], index})
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
