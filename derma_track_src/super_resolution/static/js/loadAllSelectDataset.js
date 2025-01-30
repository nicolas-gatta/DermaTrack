function addSelectOption(data, element){
    data.datasets.forEach(dataset => {
        let option = document.createElement("option");
        option.value = dataset;
        option.textContent = dataset;
        element.appendChild(option);
    });
}



function loadAllSelectDataset(){
    fetch("/super_resolution/get_datasets/0".replace(0,"training"))
    .then(response => response.json())
    .then(data => {
        addSelectOption(data, document.getElementById("train-dataset-select"));
    })

    fetch("/super_resolution/get_datasets/0".replace(0,"validation"))
    .then(response => response.json())
    .then(data => {
        addSelectOption(data, document.getElementById("valid-dataset-select"));
    })

    fetch("/super_resolution/get_datasets/0".replace(0,"evaluation"))
    .then(response => response.json())
    .then(data => {
        addSelectOption(data, document.getElementById("eval-dataset-select"));
    })
}

loadAllSelectDataset();