var annotationEnabled = false;
var annotations = [];
var drawing = false;
var canvas = document.getElementById('canvas-annotation');
var image = document.getElementById("image-preview");
var ctx = canvas.getContext('2d');
var startX, startY;
var trashButton = document.getElementById("trash-button")
var selectedAnnotation = -1;
var first_time = true;

canvas.addEventListener('contextmenu', (e) => e.preventDefault());

canvas.addEventListener('mousedown', (e) => {
    if (e.button === 2 && annotationEnabled) {
        const rect = image.getBoundingClientRect();
        startX = e.clientX - rect.left;
        startY = e.clientY - rect.top;
        drawing = true;
    }
});

canvas.addEventListener('mouseup', (e) => {
    if (drawing && annotationEnabled) {
        const rect = image.getBoundingClientRect();
        var endX = e.clientX - rect.left;
        var endY = e.clientY - rect.top;

        if (Math.abs(endX - startX) >= 0.1 && Math.abs(endY - startY) >= 0.1){
            annotations.push({ startX, startY, endX, endY });
            drawAnnotations();
        }
        drawing = false;
    }
});


async function toggleAnnotations() {
    annotationEnabled = !annotationEnabled;
    canvas.style.display = annotationEnabled ? "block" : "none";
    if(annotationEnabled){
        try {
            await fetch(`/core/visit_list/get_annotations/${canvas.dataset.id}`, {
                method: "GET",
                headers: { 
                    "Content-Type": "application/json"
                },
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'error') {
                    console.error(`Failed to update the annotation ${canvas.dataset.id}: ${data.message}`);
                }else{
                    annotations = [];
                    if (data.annotations){
                        annotations = data.annotations;
                        drawAnnotations();  
                    }
                }
            })

        } catch (error) {
            console.error(`Error Loading the annotation ${canvas.dataset.id}:`, error);
        }
    } 
}

function drawLine(startX, startY, endX, endY, color) {
    ctx.beginPath();
    ctx.moveTo(startX, startY);
    ctx.lineTo(endX, endY);
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(startX, startY, 4, 0, 2 * Math.PI);
    ctx.fillStyle = color;
    ctx.fill();

    ctx.beginPath();
    ctx.arc(endX, endY, 4, 0, 2 * Math.PI);
    ctx.fillStyle = color;
    ctx.fill();

    const pixel_length = Math.sqrt(Math.pow(endX - startX, 2) + Math.pow(endY - startY, 2)).toFixed(2);

    var length;

    if (canvas.dataset.distance != "null"){
        length = (((parseFloat(canvas.dataset.pixel_size) * pixel_length) * parseFloat(canvas.dataset.distance)) / parseFloat(canvas.dataset.focal)).toFixed(2) + " mm"
    }else{
        length = pixel_length + " px"
    }
    ctx.font = "20px Arial";
    ctx.fillStyle = color;
    ctx.fillText(length, ((startX + endX ) / 2) + 10, (startY + endY) / 2); 
}

function drawAnnotations() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    annotations.forEach(({ startX, startY, endX, endY }) => {
        drawLine(startX, startY, endX, endY, "#0af727")
    });
}


canvas.addEventListener('click', (e) => {
    const rect = canvas.getBoundingClientRect();
    var x = e.clientX - rect.left;
    var y = e.clientY - rect.top;
    var selectStartX;
    var selectStartY;
    var selectEndX;
    var SelectEndY;

    for (const annotation of annotations) {
        ctx.beginPath();
        ctx.moveTo(annotation.startX, annotation.startY);
        ctx.lineTo(annotation.endX, annotation.endY);

        ctx.lineWidth = 10;
        if (ctx.isPointInStroke(x, y)) {
            selectStartX = annotation.startX;
            selectStartY = annotation.startY;
            selectEndX = annotation.endX;
            SelectEndY = annotation.endY;
            drawLine(selectStartX, selectStartY, selectEndX, SelectEndY, "red");
            break;
        }
    }

    selectedAnnotation = annotations.findIndex(({ startX, startY, endX, endY }) =>
        startX == selectStartX && startY == selectStartY && endX == selectEndX && endY == SelectEndY
    );

    if (selectedAnnotation !== -1) {
        trashButton.style.display = 'block';
        trashButton.style.left = `${((selectStartX + selectEndX) / 2) - 40}px`;
        trashButton.style.top = `${(selectStartY + SelectEndY) / 2}px`;
    } else {
        drawAnnotations();
        trashButton.style.display = 'none';
    }
});

function deleteLine() {
    if (selectedAnnotation !== null) {
        trashButton.style.opacity = '0';
        setTimeout(() => {
            annotations.splice(selectedAnnotation, 1);
            selectedAnnotation = null;
            trashButton.style.display = 'none';
            trashButton.style.opacity = '1';
            drawAnnotations();
        }, 200);
    }
}

async function saveAnnotations(){
    try {
        await fetch(`/core/visit_list/update_visit_body_part/${canvas.dataset.id}/`, {
            method: "PUT",
            headers: { 
                "Content-Type": "application/json"
            },
            body: JSON.stringify({annotations: annotations})
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'error') {
                console.error(`Failed to update the annotation ${canvas.dataset.id}: ${data.message}`);
            }
        })

    } catch (error) {
        console.error(`Error saving the annotation ${canvas.dataset.id}:`, error);
    }
}