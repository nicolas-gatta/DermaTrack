var pretrainSelect = document.getElementById("model-select");
var architectureSelect = document.getElementById("architecture-select");
var modeSelect = document.getElementById("mode-select");
var scaleSelect = document.getElementById("scale-select");

function toggleFields(){
    var isPretrain = pretrainSelect.value !== "";
    [architectureSelect, modeSelect, scaleSelect].forEach( select => {
        select.disabled = isPretrain;
        select.required = !isPretrain;
    })
}

 pretrainSelect.addEventListener("change", toggleFields)