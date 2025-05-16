var annotationEnabled = false;
var annotations = [];
var drawing = false;
var canvas = document.getElementById('canvas-annotation');
var image = document.getElementById("image-preview");
var ctx = canvas.getContext('2d');
var startX, startY;
var trashButton = document.getElementById("trash-button")
var selectedAnnotation = -1;

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

        annotations.push({ startX, startY, endX, endY });
        drawAnnotations();
        drawing = false;
    }
});


function toggleAnnotations() {
    annotationEnabled = !annotationEnabled;
    canvas.style.display = annotationEnabled ? "block" : "none";
    drawAnnotations();
}

function drawLine(startX, startY, endX, endY, color) {
    ctx.beginPath();
    ctx.moveTo(startX, startY);
    ctx.lineTo(endX, endY);
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(startX, startY, 4, 0, 2 * Math.PI);
    ctx.fillStyle = color;
    ctx.fill();

    ctx.beginPath();
    ctx.arc(endX, endY, 4, 0, 2 * Math.PI);
    ctx.fillStyle = color;
    ctx.fill();

    const length = Math.sqrt(Math.pow(endX - startX, 2) + Math.pow(endY - startY, 2)).toFixed(2);
    ctx.font = "14px Arial";
    ctx.fillStyle = color;
    ctx.fillText(`${length}px`, ((startX + endX ) / 2) + 10, (startY + endY) / 2); 
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