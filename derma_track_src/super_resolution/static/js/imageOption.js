var subdiviseInput = document.getElementById('subdivise-radio');
var sameSizeInput = document.getElementById('resize-radio');

var subdiviseOptions = document.getElementById('subdiviseOptions');
var resizeOptions = document.getElementById('resizeOptions');

function resetOption(){
    subdiviseInput.checked = false;
    sameSizeInput.checked = false;
    toggleOptions()
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


subdiviseInput.addEventListener('change', toggleOptions);
sameSizeInput.addEventListener('change', toggleOptions);

toggleOptions();
