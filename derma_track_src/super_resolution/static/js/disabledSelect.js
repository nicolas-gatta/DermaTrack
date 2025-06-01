var pretrainSelect = document.getElementById("model-select");
var architectureSelect = document.getElementById("architecture-select");
var modeSelect = document.getElementById("mode-select");
var scaleSelect = document.getElementById("scale-select");

var subdiviseRadio = document.getElementById("subdivise-radio");
var resizeRadio = document.getElementById("resize-radio");
var patchSelect = document.getElementById("patch-size-select");
var overlayingSelect = document.getElementById("overlaying-select");
var resizeInput = document.getElementById("resize-rule-input");
var angleRadio = document.getElementById("angle-radio");
var degreeSelect = document.getElementById("degree-select");
var stepSelect = document.getElementById("step-select");


function toggleFields(){
    var isPretrain = pretrainSelect.value !== "";
    [architectureSelect, modeSelect, scaleSelect].forEach( select => {
        select.disabled = isPretrain;
        select.required = !isPretrain;
    });

    [subdiviseRadio, resizeRadio, patchSelect, overlayingSelect, resizeInput, angleRadio, degreeSelect, stepSelect].forEach( field => {
        field.disabled = isPretrain;
    });
}

 pretrainSelect.addEventListener("change", toggleFields);