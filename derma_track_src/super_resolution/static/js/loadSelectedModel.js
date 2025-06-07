async function loadSelectedModel(){

    fetch("/super_resolution/get_selected_model")
    .then(response => response.json())
    .then(data => {
        document.getElementById("selected-model").innerHTML = data.model_name;
    })
}

loadSelectedModel();