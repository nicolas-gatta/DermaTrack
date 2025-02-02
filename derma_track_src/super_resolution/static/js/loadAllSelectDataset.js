async function loadAllSelectDataset(){

    const myModule = await import("./addSelectOption.js");

    fetch("/super_resolution/get_datasets/0".replace(0,"training"))
    .then(response => response.json())
    .then(data => {
        myModule.addSelectOption(data.datasets, document.getElementById("train-dataset-select"));
    })

    fetch("/super_resolution/get_datasets/0".replace(0,"validation"))
    .then(response => response.json())
    .then(data => {
        myModule.addSelectOption(data.datasets, document.getElementById("valid-dataset-select"));
    })

    fetch("/super_resolution/get_datasets/0".replace(0,"evaluation"))
    .then(response => response.json())
    .then(data => {
        myModule.addSelectOption(data.datasets, document.getElementById("eval-dataset-select"));
    })
}

loadAllSelectDataset();