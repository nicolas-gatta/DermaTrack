var selectElement = document.getElementById("category-select");
var datasetDropdown = document.getElementById("dataset-select");

function emptyDropdown(dropdown){
    while (dropdown.options.length > 0) {
        dropdown.remove(0);
    }
}

selectElement.addEventListener("change", (event) => {
  emptyDropdown(datasetDropdown);
  datasetDropdown.append('<option value="">Loading...</option>');
  if (event.target.value) {
    fetch("get_datasets/0".replace(0,event.target.value))
        .then(response => response.json())
        .then(data => {
            emptyDropdown(datasetDropdown);
            let defaultOption = document.createElement("option");
            defaultOption.value = "";
            defaultOption.textContent = "Select Dataset";
            datasetDropdown.appendChild(defaultOption);
            data.datasets.forEach(dataset => {
                let option = document.createElement("option");
                option.value = dataset;
                option.textContent = dataset;
                datasetDropdown.appendChild(option);
            });
        })
        .catch(error => {
            emptyDropdown(datasetDropdown);
            let errorOption = document.createElement("option");
            errorOption.textContent = "Error loading datasets";
            datasetDropdown.appendChild(errorOption);
        });
    } else {
        emptyDropdown(datasetDropdown);
        let defaultOption = document.createElement("option");
        defaultOption.textContent = "Select Dataset";
        datasetDropdown.appendChild(defaultOption);
    }
  })



