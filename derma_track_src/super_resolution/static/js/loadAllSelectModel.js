async function loadAllSelectModel(){

    const myModule = await import("./addSelectOption.js");

    fetch("/super_resolution/get_models")
    .then(response => response.json())
    .then(data => {
        myModule.addSelectOption(data.models, document.getElementById("model-select"));
    })
}

loadAllSelectModel();