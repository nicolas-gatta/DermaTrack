var port;
var reader;
var dataReading = false;
var writer;
var visitId;

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
    fetch(`/camera/get_body_parts/`, {
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
    visitId = document.getElementById("mainModal").dataset.visitId;
    localStorage.setItem("capturedImages", JSON.stringify([]));
    updateCarousel([]);
    //let deviceId = await getCameraDeviceIdByName("Arducam IMX179 8MP Camera");
    if (true || await isCameraConnected(deviceId)){
        let video = document.getElementById("stream");
        video.muted = true;
        navigator.mediaDevices.getUserMedia({video: {width: { ideal:  1920} , height: { ideal: 1080 }}})
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
        console.error("Failed to connect to the serial port (Only Work on Chrome):", error);
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
        try{
            await reader.cancel();
        }catch(e){
            console.warn("Error while canceling reader:", e);
        }
        reader.releaseLock();
        
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
                        console.log(data);
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
        fetch("/camera/detect", {
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
    let selectBodyPart = document.getElementById("body-part");
    let bodyPart = selectBodyPart.selectedOptions[0].value;

    if (!bodyPart) {
        alert("Please select a body part before capturing the image.");
        selectBodyPart.classList.add("is-invalid");
        return;
    } else {
        selectBodyPart.classList.remove("is-invalid");
    }
    let video = document.getElementById("stream");
    let canvas = document.createElement("canvas");
    canvas.width = 1920;
    canvas.height = 1080;
    let context = canvas.getContext("2d");
    

    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    imageUrl = canvas.toDataURL("image/png");
    if (isSaved){
        await saveImageToServer(imageUrl, bodyPart, distance);
    }else{
        return imageUrl.split(",")[1];
    }
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

async function createVisitBodyPart(bodyPartId, distance){
    try{
        const response = await fetch(`/camera/create_visit_body_part/`,{
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "X-CSRFToken": csrftoken
            },
            body: JSON.stringify({visitId: visitId, bodyPartId: bodyPartId, distance :distance, pixelSize: 0.0014})
        })

        data = response.json();

        if (data.status === 'error') {
            console.error(`Failed to create the visit body part: ${data.message}`);
        }else{
            return data;
        }
    }catch (error) {
        console.error(`Error create the visit body part :`, error);
        return null;
    }
}

async function saveImageWithoutUpdate(visit_body_part_id, visitId, bodyPart, imageUrl, imageWidth, imageHeigth, index){
    try{
        await fetch(`/camera/save_image_without_db_update/`, {
            method: "POST",
            headers: { 
                "Content-Type": "application/json",
                "X-CSRFToken": csrftoken
            },
            body: JSON.stringify({visit_body_part_id: visit_body_part_id, visit: visitId, body_part: bodyPart, image: imageUrl, imageWidth: imageWidth, imageHeigth: imageHeigth, index})
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'error') {
                console.error(`Failed to save image ${index}: ${data.message}`);
            }
        });
    } catch (error) {
        console.error(`Error saving image ${index + 1}:`, error);
        return null;
    }
}


async function saveImageWithUpdate(visit_body_part_id, imageUrl, imageWidth, imageHeigth, index){
    try{
        await fetch(`/camera/save_image_with_db_update/`, {
            method: "POST",
            headers: { 
                "Content-Type": "application/json",
                "X-CSRFToken": csrftoken
            },
            body: JSON.stringify({visit_body_part_id: visit_body_part_id, image: imageUrl, imageWidth: imageWidth, imageHeigth: imageHeigth, index})
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'error') {
                console.error(`Failed to save image ${index}: ${data.message}`);
            }else{
                let storedImages = JSON.parse(localStorage.getItem("capturedImages"));
                storedImages.push("/media/visits/visit_"+data.visitId+"/"+data.bodyPart+"/"+data.image);
                updateCarousel(storedImages);
                localStorage.setItem("capturedImages", JSON.stringify(storedImages));
            }
        })
    } catch (error) {
        console.error(`Error saving image ${index + 1}:`, error);
    }
}

async function saveImageToServer(imageUrl, bodyPartId, distance){
    
    var data = await createVisitBodyPart(bodyPartId, distance); 

    for (let index = 0; index < 5; index++){
        if (index !== 2){
            saveImageWithoutUpdate(data.visit_body_part_id, data.visitId, data.bodyPart, imageUrl, 1920, 1080, index);
        }else{
            saveImageWithUpdate(data.visit_body_part_id, imageUrl, 1920, 1080, index);
        }
    }
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
        saveImageToServer(visitId, bodyPartImages[index], distanceImages[index], index);
    });

    document.getElementById("save-picture").textContent = "Save all Pictures";
}