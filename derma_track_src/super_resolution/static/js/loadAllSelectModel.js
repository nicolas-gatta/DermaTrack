async function loadAllSelectModel(){

    const myModule = await import("./addSelectOption.js");

    fetch("/super_resolution/get_models")
    .then(response => response.json())
    .then(data => {
        if (document.getElementById("evaluation-form") || document.getElementById("select-model-form")){
            data.models.push("BICUBIC_x2");
            data.models.push("BICUBIC_x4");
        }
        myModule.addSelectOption(data.models, document.getElementById("model-select"));
    })
}

loadAllSelectModel();