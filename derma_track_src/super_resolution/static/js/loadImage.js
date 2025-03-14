var imageSelector = document.getElementById("image-select");
var scaleSelector = document.getElementById("scale-select");
var modelSelector = document.getElementById("model-select");
var hrCanvas = document.getElementById("hr-canvas");
var lrCanvas = document.getElementById("lr-canvas");
var reconstructedCanvas = document.getElementById("reconstructed-canvas");

function drawImageOnCanvas(canvas, imageUrl) {
    let cacheBuster = new Date().getTime();
    let ctx = canvas.getContext("2d");
    let img = new Image();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    img.onload = function() {
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    };
    img.src = `${imageUrl}?t=${cacheBuster}`;
}

imageSelector.addEventListener("change", (event) => {

    fetch("/super_resolution/get_test_image/0".replace(0, event.target.value))
    .then(response => response.json())
    .then(data => {
        drawImageOnCanvas(hrCanvas, data.url);
    })

})

scaleSelector.addEventListener("change", (event) => {

    if(imageSelector.value != ""){
        fetch("/super_resolution/degrade_and_save_image/0/1".replace(0, imageSelector.value).replace(1, event.target.value))
        .then(response => response.json())
        .then(data => {
            drawImageOnCanvas(lrCanvas, data.url);
        })
    }
})


modelSelector.addEventListener("change", (event) => {

    if(modelSelector.value != ""){
        fetch("/super_resolution/load_test_model/0".replace(0, event.target.value), {
            method: "POST",
            headers: { 
                "Content-Type": "application/json",
                "X-CSRFToken": csrftoken
            }
        })
    }
})

function test_model() {
    if(modelSelector.value != ""){
        fetch("/super_resolution/apply_test_sr/0".replace(0, imageSelector.value))
        .then(response => response.json())
        .then(data => {
            drawImageOnCanvas(reconstructedCanvas, data.url);
        })
    }
}


