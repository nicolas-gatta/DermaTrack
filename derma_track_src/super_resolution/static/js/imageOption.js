var subdiviseInput = document.getElementById('subdivise-radio');
var sameSizeInput = document.getElementById('resize-radio');
var angleInput = document.getElementById('angle-radio');

var subdiviseOptions = document.getElementById('subdiviseOptions');
var resizeOptions = document.getElementById('resizeOptions');
var angleOptions = document.getElementById('angleOptions');

function resetOption(){
    subdiviseInput.checked = false;
    sameSizeInput.checked = false;
    toggleOptions();
}

function resetOptionPlus(){
    angleInput.checked = false;
    toggleOptionPlus();
}

function toggleOptions(){
    if (subdiviseInput.checked) {
        subdiviseOptions.classList.remove('d-none');
        resizeOptions.classList.add('d-none');

    } else if (sameSizeInput.checked) {
        subdiviseOptions.classList.add('d-none');
        resizeOptions.classList.remove('d-none');

    } else {
        subdiviseOptions.classList.add('d-none');
        resizeOptions.classList.add('d-none');
    }
}

function toggleOptionPlus(){
    if (angleInput.checked) {
        angleOptions.classList.remove('d-none');

    } else {
        angleOptions.classList.add('d-none');
    }
}


subdiviseInput.addEventListener('change', toggleOptions);
sameSizeInput.addEventListener('change', toggleOptions);
angleInput.addEventListener('change', toggleOptionPlus);

toggleOptions();
toggleOptionPlus();
