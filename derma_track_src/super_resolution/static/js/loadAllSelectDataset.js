async function loadAllSelectDataset(){

    const myModule = await import("./addSelectOption.js");

    trainDataset = document.getElementById("train-dataset-select");
    validDataset = document.getElementById("valid-dataset-select");
    evalDataset = document.getElementById("eval-dataset-select")

    if (trainDataset){
        fetch("/super_resolution/get_datasets/0".replace(0,"training"))
        .then(response => response.json())
        .then(data => {
            myModule.addSelectOption(data.datasets, trainDataset);
        })
    }

    if (validDataset){
        fetch("/super_resolution/get_datasets/0".replace(0,"validation"))
        .then(response => response.json())
        .then(data => {
            myModule.addSelectOption(data.datasets, validDataset);
        })
    }

    if (evalDataset){
        fetch("/super_resolution/get_datasets/0".replace(0,"evaluation"))
        .then(response => response.json())
        .then(data => {
            myModule.addSelectOption(data.datasets, evalDataset);
        })
    }
}

loadAllSelectDataset();